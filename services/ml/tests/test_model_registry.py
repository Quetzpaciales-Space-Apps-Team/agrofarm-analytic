"""ML-050: models live under <name>/<version>/ with a complete card, and a deploy that asks for a
model refuses to start when it is missing, empty or not the version the card names."""

import json
from pathlib import Path
from typing import Any

import pytest

from agrofarm_ml import registry
from agrofarm_ml.main import create_app
from agrofarm_ml.registry import REQUIRED_FIELDS, RegistryError

ML_ROOT = Path(__file__).resolve().parents[1]
CARDS = ML_ROOT / "registry"
EXPERIMENTS = ML_ROOT.parents[1] / "experiments"
REQUIRED_MODELS = "maize-leaf/0.1.1,ndvi-outlook/0.1.0"


def card(name: str = "toy", version: str = "1.0.0", **changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "version": version,
        "model_version": f"{name}-{version}",
        "status": "experimental",
        "purpose": "A toy model for the tests.",
        "label": "toy_label",
        "training_data_versions": {"source": "fixture"},
        "features": ["a", "b"],
        "held_out_metrics": {"split": "whole years", "skill": 0.1},
        "limits": ["Only a toy."],
    }
    return {**base, **changes}


def install(
    root: Path,
    name: str = "toy",
    version: str = "1.0.0",
    registry_card: dict[str, Any] | None = None,
    artifact: tuple[str, dict[str, Any]] | None = (
        "model_card.json",
        {"model_version": "toy-1.0.0"},
    ),
) -> Path:
    directory = root / name / version
    directory.mkdir(parents=True)
    (directory / "registry_card.json").write_text(
        json.dumps(registry_card if registry_card is not None else card(name, version))
    )
    if artifact is not None:
        (directory / artifact[0]).write_text(json.dumps(artifact[1]))
    return directory


# --- the cards that ship with the service -------------------------------------------------


def test_the_repository_registry_describes_both_runtime_models() -> None:
    entries = registry.scan(CARDS)
    assert {f"{e.name}/{e.version}" for e in entries} == set(REQUIRED_MODELS.split(","))


def test_every_shipped_card_has_every_required_field_filled() -> None:
    for entry in registry.scan(CARDS):
        for field in REQUIRED_FIELDS:
            assert entry.card[field], (entry.name, field)


def test_the_leaf_card_agrees_with_the_experiment_card() -> None:
    source = json.loads((EXPERIMENTS / "2026-11-15-maize-leaf/model_card.json").read_text())
    entry = next(e for e in registry.scan(CARDS) if e.name == "maize-leaf")
    assert entry.card["model_version"] == source["model_version"]
    assert entry.card["training_data_versions"] == source["data_versions"]
    assert entry.card["limits"] == source["limits"]
    assert entry.card["held_out_metrics"]["macro_f1"] == source["test_metrics"]["macro_f1"]
    assert entry.card["held_out_metrics"]["rows"] == source["test_metrics"]["rows"]
    assert entry.card["status"] == source["status"]


def test_the_vegetation_card_agrees_with_the_experiment_card_and_keeps_the_failed_goal() -> None:
    source = json.loads(
        (EXPERIMENTS / "2026-11-15-label-search/ndvi_peak_model_card.json").read_text()
    )
    entry = next(e for e in registry.scan(CARDS) if e.name == "ndvi-outlook")
    assert entry.card["model_version"] == "ndvi-vegetation-outlook-0.1.0"
    assert entry.card["training_data_versions"] == source["data_versions"]
    assert entry.card["features"] == source["features"]
    assert entry.card["limits"] == source["limits"]
    metrics = entry.card["held_out_metrics"]
    assert metrics["pooled_skill"] == source["metrics"]["pooled_skill"]
    assert metrics["years"] == source["metrics"]["years"]
    assert metrics["verdict"] == source["verdict"] == "no_skill_over_climatology"


# --- layout and card rules -----------------------------------------------------------------


def test_scan_reads_name_version_directories(tmp_path: Path) -> None:
    install(tmp_path)
    install(tmp_path, "other", "2.0.0")
    assert {(e.name, e.version) for e in registry.scan(tmp_path)} == {
        ("toy", "1.0.0"),
        ("other", "2.0.0"),
    }


def test_a_card_must_match_the_directory_it_sits_in(tmp_path: Path) -> None:
    install(tmp_path, registry_card=card("toy", "9.9.9"))
    with pytest.raises(RegistryError, match="toy/1.0.0"):
        registry.scan(tmp_path)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_card_missing_a_required_field_is_refused_and_names_it(
    tmp_path: Path, field: str
) -> None:
    incomplete = card()
    del incomplete[field]
    install(tmp_path, registry_card=incomplete)
    with pytest.raises(RegistryError, match=field):
        registry.scan(tmp_path)


@pytest.mark.parametrize("field", ["limits", "features", "purpose", "held_out_metrics"])
def test_an_empty_required_field_counts_as_missing(tmp_path: Path, field: str) -> None:
    empty = card()
    empty[field] = [] if field in {"limits", "features"} else ""
    install(tmp_path, registry_card=empty)
    with pytest.raises(RegistryError, match=field):
        registry.scan(tmp_path)


# --- require ------------------------------------------------------------------------------


def test_require_accepts_a_model_whose_artifact_card_names_the_same_version(
    tmp_path: Path,
) -> None:
    install(tmp_path)
    assert [e.name for e in registry.require(tmp_path, ["toy/1.0.0"])] == ["toy"]


def test_require_prefers_the_serving_card_when_the_directory_has_one(tmp_path: Path) -> None:
    directory = install(tmp_path, artifact=("model_card.json", {"model_version": "older-model"}))
    (directory / "serving_card.json").write_text(json.dumps({"model_version": "toy-1.0.0"}))
    registry.require(tmp_path, ["toy/1.0.0"])


def test_require_names_a_model_that_is_not_installed(tmp_path: Path) -> None:
    install(tmp_path)
    with pytest.raises(RegistryError, match=r"missing/0\.1\.0"):
        registry.require(tmp_path, ["toy/1.0.0", "missing/0.1.0"])


def test_require_refuses_a_directory_without_an_artifact_card(tmp_path: Path) -> None:
    install(tmp_path, artifact=None)
    with pytest.raises(RegistryError, match="artifact card"):
        registry.require(tmp_path, ["toy/1.0.0"])


def test_require_refuses_an_artifact_that_is_not_the_version_the_card_names(
    tmp_path: Path,
) -> None:
    install(tmp_path, artifact=("model_card.json", {"model_version": "toy-0.0.1"}))
    with pytest.raises(RegistryError, match="toy-0.0.1"):
        registry.require(tmp_path, ["toy/1.0.0"])


def test_require_with_an_unreadable_card_says_so(tmp_path: Path) -> None:
    directory = install(tmp_path)
    (directory / "registry_card.json").write_text("{not json")
    with pytest.raises(RegistryError, match="toy/1.0.0"):
        registry.require(tmp_path, ["toy/1.0.0"])


# --- the service ----------------------------------------------------------------------------


def test_the_service_refuses_to_start_when_a_required_model_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUIRE_MODELS", "toy/1.0.0")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    with pytest.raises(RegistryError, match="toy/1.0.0"):
        create_app()


def test_the_service_starts_when_the_required_model_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(tmp_path)
    monkeypatch.setenv("REQUIRE_MODELS", "toy/1.0.0")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    assert create_app() is not None


def test_nothing_is_required_unless_the_deploy_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUIRE_MODELS", raising=False)
    monkeypatch.setenv("MODEL_REGISTRY_DIR", "/does/not/exist")
    assert create_app() is not None


@pytest.mark.parametrize(
    "value", ["toy/1.0.0;other/2.0.0", "toy/1.0.0,other/2.0.0", " toy/1.0.0 ; "]
)
def test_the_required_list_accepts_semicolons_because_gcloud_splits_on_commas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    install(tmp_path)
    install(
        tmp_path, "other", "2.0.0", artifact=("model_card.json", {"model_version": "other-2.0.0"})
    )
    monkeypatch.setenv("REQUIRE_MODELS", value)
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    assert create_app() is not None
    monkeypatch.setenv("REQUIRE_MODELS", "toy/1.0.0;absent/0.0.1")
    with pytest.raises(RegistryError, match=r"absent/0\.0\.1"):
        create_app()
