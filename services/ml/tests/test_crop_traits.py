"""ML-038: the crop traits table is versioned data, validated, and honest about its status."""

import copy
import json
from typing import Any

import pytest

from agrofarm_ml.rotations.traits import (
    PRIORITY_TRAITS,
    SCALE_TRAITS,
    TraitsError,
    TraitsTable,
    load_traits,
    validate_table,
)

PILOT_CROPS = {"maize", "black_beans", "cover_crop", "fallow"}


def raw() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = load_traits()
    return copy.deepcopy(table.manifest), copy.deepcopy(table.rows)


def test_one_row_per_pilot_crop() -> None:
    table = load_traits()
    assert {row["crop_id"] for row in table.rows} == PILOT_CROPS
    assert len(table.rows) == len(PILOT_CROPS)


def test_manifest_preserves_source_status_and_records_owner_confirmed_review() -> None:
    manifest = load_traits().manifest
    assert manifest["version"] == "0.1.0"
    assert manifest["status"] == "provisional"
    assert manifest["reviewed_by"] is None
    review = manifest["human_review_confirmation"]
    assert review["confirmed_by"] == "eldmark"
    assert review["confirmed_on"] == "2026-11-15"
    assert review["review_occurred"] is True
    assert review["reviewer_identity"] is None
    assert manifest["values_verified_against_sources"] is False
    assert "claude-sonnet-5-5" in manifest["generated_by"]
    assert manifest["sources"], "the manifest lists the source catalogue"


def test_every_row_has_every_trait_the_issue_asks_for() -> None:
    required = {
        "cycle_days",
        "kc_mid",
        "root_depth_m",
        "soil_cover",
        "water_demand",
        "nitrogen_fixing",
        "forage_value",
        "market_suitability",
        *PRIORITY_TRAITS,
    }
    for row in load_traits().rows:
        assert required <= set(row["values"]), row["crop_id"]
        assert isinstance(row["aliases"], list)


def test_every_value_cites_a_source_in_the_manifest_with_a_basis_text() -> None:
    table = load_traits()
    catalogue = set(table.manifest["sources"])
    for row in table.rows:
        assert set(row["sources"]) == set(row["values"]), row["crop_id"]
        for trait, cite in row["sources"].items():
            assert cite["source_id"] in catalogue, (row["crop_id"], trait)
            assert len(cite["basis"]) >= 10, (row["crop_id"], trait)


def test_scale_traits_stay_inside_zero_to_one() -> None:
    for row in load_traits().rows:
        for trait in SCALE_TRAITS:
            assert 0.0 <= row["values"][trait] <= 1.0, (row["crop_id"], trait)


@pytest.mark.parametrize("trait", SCALE_TRAITS)
@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf")])
def test_scale_traits_reject_booleans_and_nonfinite_numbers(trait: str, value: Any) -> None:
    manifest, rows = raw()
    rows[0]["values"][trait] = value
    with pytest.raises(TraitsError, match=trait):
        validate_table(manifest, rows)


@pytest.mark.parametrize("trait", ["cycle_days", "kc_mid", "root_depth_m"])
@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf")])
def test_physical_traits_reject_booleans_and_nonfinite_numbers(trait: str, value: Any) -> None:
    manifest, rows = raw()
    rows[0]["values"][trait] = value
    with pytest.raises(TraitsError, match=trait):
        validate_table(manifest, rows)


def test_aliases_resolve_to_the_crop_and_are_unique() -> None:
    table = load_traits()

    def crop_id(name: str) -> str | None:
        crop = table.resolve(name)
        return crop.crop_id if crop else None

    assert crop_id("Maize ") == "maize"
    assert crop_id("black beans") == "black_beans"
    assert crop_id("corn") == "maize"
    assert table.resolve("pumpkin") is None
    seen: list[str] = []
    for row in table.rows:
        seen += [row["crop_id"], *row["aliases"]]
    assert len(seen) == len(set(seen))


def test_aliases_are_english_only() -> None:
    for row in load_traits().rows:
        for alias in row["aliases"]:
            assert alias.isascii(), alias


def test_the_packaged_table_is_valid() -> None:
    manifest, rows = raw()
    validate_table(manifest, rows)  # does not raise


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m, r: m.pop("version"), "version"),
        (lambda m, r: m.update(status="reviewed"), "status"),
        (lambda m, r: r[0]["values"].update(income=1.5), "income"),
        (lambda m, r: r[0]["values"].pop("kc_mid"), "kc_mid"),
        (lambda m, r: r[0]["sources"].pop("income"), "income"),
        (lambda m, r: r[0]["sources"]["income"].update(source_id="nope"), "nope"),
        (lambda m, r: r[0]["sources"]["income"].update(basis=""), "basis"),
        (lambda m, r: r.append(copy.deepcopy(r[0])), "duplicate"),
        (lambda m, r: r[0].update(aliases="maiz"), "aliases"),
    ],
    ids=[
        "no_version",
        "status",
        "out_of_scale",
        "missing_trait",
        "missing_source",
        "unknown_source",
        "empty_basis",
        "duplicate_crop",
        "aliases_not_list",
    ],
)
def test_schema_rejects_broken_tables(mutate: Any, message: str) -> None:
    manifest, rows = raw()
    mutate(manifest, rows)
    with pytest.raises(TraitsError, match=message):
        validate_table(manifest, rows)


def test_a_reviewed_table_must_name_its_reviewer() -> None:
    manifest, rows = raw()
    manifest["status"] = "reviewed"
    with pytest.raises(TraitsError, match="reviewed_by"):
        validate_table(manifest, rows, allow_reviewed=True)
    manifest["reviewed_by"] = "a local agronomist"
    validate_table(manifest, rows, allow_reviewed=True)


def test_files_on_disk_are_what_the_loader_reads() -> None:
    from importlib import resources

    base = resources.files("agrofarm_ml") / "rotations" / "data"
    manifest = json.loads((base / "manifest.json").read_text("utf-8"))
    assert manifest["version"] == load_traits().manifest["version"]
    assert isinstance(load_traits(), TraitsTable)
