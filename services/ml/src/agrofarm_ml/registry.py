"""Model registry: every model lives under `<root>/<name>/<version>/` with a complete card.

Layout of one model directory (the same tree in the image, in `services/ml/registry/` for the
cards, and in Cloud Storage under `models/<name>/<version>/` when an operator publishes it):

    registry_card.json   the card this module validates (fields in REQUIRED_FIELDS)
    serving_card.json    the card of the artifact itself, when the model has one; else
    model_card.json      the card the training code wrote
    <model files>        read and checksummed by the model's own `load()`

A deploy that needs a model sets `REQUIRE_MODELS=name/version,...`. `require()` then raises
`RegistryError` at start-up if a model is missing, its card is incomplete, or its artifact is not
the version the card names, so a deploy without a valid model never takes traffic. With nothing
required (local runs, tests) the service starts as before and the routes of a model that is not
configured answer 503.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CARD_FILE = "registry_card.json"
ARTIFACT_CARDS = ("serving_card.json", "model_card.json")  # the first one that exists is used
REQUIRED_FIELDS = (
    "name",
    "version",
    "model_version",
    "status",
    "purpose",
    "label",
    "training_data_versions",
    "features",
    "held_out_metrics",
    "limits",
)


class RegistryError(RuntimeError):
    """A model is missing or its card cannot be trusted."""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    version: str
    directory: Path
    card: dict[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(
            f"{label}: {path} cannot be read as JSON ({type(exc).__name__})"
        ) from exc
    if not isinstance(parsed, dict):
        raise RegistryError(f"{label}: {path} must hold a JSON object")
    return parsed


def _entry(directory: Path, name: str, version: str) -> ModelEntry:
    label = f"{name}/{version}"
    card = _read_json(directory / CARD_FILE, label)
    for field in REQUIRED_FIELDS:
        if field not in card or card[field] in (None, "", [], {}):
            raise RegistryError(f"{label}: the card has no value for '{field}'")
    if card["name"] != name or card["version"] != version:
        raise RegistryError(
            f"{label}: the card says {card['name']}/{card['version']}, which is not its directory"
        )
    return ModelEntry(name, version, directory, card)


def scan(root: Path) -> list[ModelEntry]:
    """Every `<root>/<name>/<version>/registry_card.json`, validated."""
    entries = []
    for card_path in sorted(root.glob(f"*/*/{CARD_FILE}")):
        directory = card_path.parent
        entries.append(_entry(directory, directory.parent.name, directory.name))
    return entries


def _artifact_version(entry: ModelEntry) -> str:
    label = f"{entry.name}/{entry.version}"
    for name in ARTIFACT_CARDS:
        path = entry.directory / name
        if path.is_file():
            return str(_read_json(path, label).get("model_version", ""))
    raise RegistryError(
        f"{label}: the directory has no artifact card ({' or '.join(ARTIFACT_CARDS)})"
    )


def require(root: Path, required: Iterable[str]) -> list[ModelEntry]:
    """Each listed `name/version` model, present, with a complete card and the right artifact."""
    entries = []
    for item in required:
        name, _, version = item.partition("/")
        directory = root / name / version
        if not name or not version or not (directory / CARD_FILE).is_file():
            raise RegistryError(f"{item}: required model not found under {root}")
        entry = _entry(directory, name, version)
        artifact = _artifact_version(entry)
        if artifact != entry.card["model_version"]:
            raise RegistryError(
                f"{item}: the card names {entry.card['model_version']} but the artifact is "
                f"{artifact or 'unversioned'}"
            )
        entries.append(entry)
    return entries
