"""Versioned crop traits table for the pilot crops (ML-038, ADR 0004).

Crop traits live in packaged data (`data/crop_traits.json` plus `data/manifest.json`), not in
code. Source status remains PROVISIONAL; the manifest separately records owner-confirmed review.
This module validates the schema and resolves crop names and aliases. It does no scoring.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

# The seven farmer priorities (the keys of the `scores` object in the API contract).
PRIORITIES = (
    "water_use",
    "income",
    "soil_structure",
    "forage_value",
    "nitrogen_fixing",
    "erosion_reduction",
    "market_suitability",
)
# Traits scored 0..1 that the rules read. `water_demand` feeds the `water_use` priority.
SCALE_TRAITS = (
    "soil_cover",
    "water_demand",
    "nitrogen_fixing",
    "forage_value",
    "market_suitability",
    "income",
    "soil_structure",
    "erosion_reduction",
)
PRIORITY_TRAITS = tuple(trait for trait in SCALE_TRAITS if trait != "water_demand")
# Physical traits that may be null when a row is a period (fallow) or a species is not chosen.
NULLABLE_TRAITS = ("cycle_days", "kc_mid", "root_depth_m")
ALL_TRAITS = (*NULLABLE_TRAITS, *SCALE_TRAITS)
GROUPS = ("cereal", "legume", "cover", "fallow")
MANIFEST_KEYS = ("name", "version", "status", "generated_by", "reviewed_by", "sources")


class TraitsError(ValueError):
    """The crop traits table does not match its schema."""


def normalize(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


@dataclass(frozen=True)
class Crop:
    crop_id: str
    display_name: str
    group: str
    values: dict[str, float | None]
    sources: dict[str, dict[str, str]]

    def trait(self, name: str) -> float:
        value = self.values[name]
        if value is None:  # only the physical traits are nullable and rules never read them
            raise TraitsError(f"{self.crop_id}.{name} has no value")
        return value


@dataclass(frozen=True)
class TraitsTable:
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    def resolve(self, name: str) -> Crop | None:
        key = normalize(name)
        for row in self.rows:
            if key == row["crop_id"] or key in row["aliases"]:
                return Crop(
                    crop_id=row["crop_id"],
                    display_name=row["display_name"],
                    group=row["group"],
                    values=dict(row["values"]),
                    sources={k: dict(v) for k, v in row["sources"].items()},
                )
        return None


def validate_table(
    manifest: dict[str, Any], rows: list[dict[str, Any]], *, allow_reviewed: bool = False
) -> None:
    """Raise TraitsError when the table breaks its schema. Reviewed status needs a reviewer."""
    for key in MANIFEST_KEYS:
        if key not in manifest:
            raise TraitsError(f"manifest is missing {key}")
    allowed = {"provisional", "reviewed"} if allow_reviewed else {"provisional"}
    if manifest["status"] not in allowed:
        raise TraitsError(f"manifest status must be one of {sorted(allowed)}")
    if manifest["status"] == "reviewed" and not manifest["reviewed_by"]:
        raise TraitsError("a reviewed table must name reviewed_by")
    catalogue = manifest["sources"]
    seen: set[str] = set()
    for row in rows:
        crop = row.get("crop_id", "?")
        names = [crop, *row["aliases"]] if isinstance(row.get("aliases"), list) else None
        if names is None:
            raise TraitsError(f"{crop}: aliases must be a list")
        if any(name in seen for name in names):
            raise TraitsError(f"{crop}: duplicate crop id or alias")
        seen.update(names)
        if row.get("group") not in GROUPS:
            raise TraitsError(f"{crop}: group must be one of {GROUPS}")
        _validate_values(crop, row["values"])
        _validate_sources(crop, row["values"], row["sources"], catalogue)


def _validate_values(crop: str, values: dict[str, Any]) -> None:
    for trait in ALL_TRAITS:
        if trait not in values:
            raise TraitsError(f"{crop}: missing trait {trait}")
    for trait in SCALE_TRAITS:
        value = values[trait]
        if not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
            raise TraitsError(f"{crop}: {trait} must be a number from 0 to 1")
    for trait in NULLABLE_TRAITS:
        value = values[trait]
        if value is not None and (not isinstance(value, int | float) or value <= 0):
            raise TraitsError(f"{crop}: {trait} must be positive or null")


def _validate_sources(
    crop: str, values: dict[str, Any], sources: dict[str, Any], catalogue: dict[str, Any]
) -> None:
    for trait in values:
        cite = sources.get(trait)
        if cite is None:
            raise TraitsError(f"{crop}: {trait} has no source")
        if cite.get("source_id") not in catalogue:
            raise TraitsError(f"{crop}: {trait} cites unknown source {cite.get('source_id')}")
        if len(str(cite.get("basis", "")).strip()) < 10:
            raise TraitsError(f"{crop}: {trait} needs a basis text")


@lru_cache(maxsize=1)
def load_traits() -> TraitsTable:
    base = resources.files("agrofarm_ml") / "rotations" / "data"
    manifest = json.loads((base / "manifest.json").read_text("utf-8"))
    rows = json.loads((base / "crop_traits.json").read_text("utf-8"))
    validate_table(manifest, rows)
    return TraitsTable(manifest=manifest, rows=rows)
