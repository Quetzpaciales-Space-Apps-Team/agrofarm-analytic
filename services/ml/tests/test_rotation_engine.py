"""ML-040: the engine scores at least three strategies and explains every move of a score."""

from typing import Any

import pytest

from agrofarm_ml.rotations import engine
from agrofarm_ml.rotations.rules import PRIORITIES

CALM = {"rainfall_change_pct": 0.0, "temperature_change_c": 0.0, "enso_phase": "neutral"}
DRY = {"rainfall_change_pct": -30.0, "temperature_change_c": 2.0, "enso_phase": "el_nino"}
EQUAL = dict.fromkeys(PRIORITIES, 0.5)


def run(
    current: list[str],
    scenario: dict[str, Any] = CALM,
    weights: dict[str, float] | None = None,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chosen = weights if weights is not None else EQUAL
    return engine.score(current, chosen, constraints or {}, scenario)


def by_id(strategies: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in strategies}


def test_at_least_three_strategies_and_exactly_one_is_current() -> None:
    strategies = run(["maize", "fallow"])
    assert len(strategies) >= 3
    assert [s["is_current"] for s in strategies].count(True) == 1
    assert strategies[0]["id"] == "current"


def test_a_current_rotation_that_is_a_candidate_is_not_listed_twice() -> None:
    strategies = run(["maize", "black_beans"])
    sequences = [tuple(s["sequence"]) for s in strategies]
    assert len(sequences) == len(set(sequences)) >= 3


def test_aliases_are_understood_in_the_current_rotation() -> None:
    strategies = run(["Corn", "beans"])
    assert strategies[0]["sequence"] == ["maize", "black_beans"]


@pytest.mark.parametrize("scenario", [CALM, DRY])
def test_every_strategy_has_the_full_decision_support_fields(scenario: dict[str, Any]) -> None:
    for strategy in run(["maize", "fallow"], scenario):
        assert strategy["why"] and strategy["evidence"] and strategy["benefits"]
        assert strategy["gaps"], "output without gaps is rejected"
        assert strategy["risks"], "output without risks is rejected"
        assert strategy["needs_local_review"] is True
        assert all(r["needs_local_review"] is True for r in strategy["risks"])
        assert set(strategy["scores"]) == set(PRIORITIES)
        low, high = strategy["uncertainty"]["low"], strategy["uncertainty"]["high"]
        assert 0 <= low <= strategy["overall_score"] <= high <= 1


def test_the_gaps_say_the_traits_are_provisional_and_need_an_agronomist() -> None:
    for strategy in run(["maize", "fallow"]):
        text = " ".join(strategy["gaps"]).lower()
        assert "provisional" in text
        assert "local agronomist" in text


def test_evidence_sources_point_at_traits_rows_or_named_inputs() -> None:
    allowed_prefixes = ("crop_traits:", "scenario_request", "rule_constant:", "farm_profile")
    for strategy in run(["maize", "black_beans"], DRY):
        sources = [e["source"] for e in strategy["evidence"]]
        assert any(s.startswith("crop_traits:") for s in sources)
        assert all(s.startswith(allowed_prefixes) for s in sources), sources
    row_ids = {e["source"] for s in run(["maize"]) for e in s["evidence"] if ":" in e["source"]}
    assert "crop_traits:0.1.0:maize" in row_ids


def test_a_dry_scenario_lowers_water_use_and_the_response_names_the_rule() -> None:
    calm = by_id(run(["maize", "fallow"]))
    dry = by_id(run(["maize", "fallow"], DRY))
    for key in ("maize_beans", "current"):
        assert dry[key]["scores"]["water_use"] < calm[key]["scores"]["water_use"]
    text = " ".join(dry["maize_beans"]["why"])
    assert "dry_scenario_water_fit" in text
    assert "enso_rainfall_equivalent" in text
    assert "dry_scenario_water_fit" not in " ".join(calm["maize_beans"]["why"])


def test_the_scenario_effect_is_modest() -> None:
    calm = by_id(run(["maize", "fallow"]))
    dry = by_id(run(["maize", "fallow"], DRY))
    for key, strategy in dry.items():
        assert abs(strategy["overall_score"] - calm[key]["overall_score"]) < 0.1


def test_a_stronger_scenario_widens_the_uncertainty() -> None:
    def width(s: dict[str, Any]) -> float:
        return float(s["uncertainty"]["high"] - s["uncertainty"]["low"])

    assert width(run(["maize", "fallow"], DRY)[0]) > width(run(["maize", "fallow"])[0])


def test_wet_scenario_hurts_erosion_scores() -> None:
    wet = {"rainfall_change_pct": 50.0, "temperature_change_c": 0.0, "enso_phase": "la_nina"}
    calm = by_id(run(["maize", "fallow"]))
    rainy = by_id(run(["maize", "fallow"], wet))
    assert (
        rainy["current"]["scores"]["erosion_reduction"]
        < calm["current"]["scores"]["erosion_reduction"]
    )
    assert "wet_scenario_erosion_exposure" in " ".join(rainy["current"]["why"])


def test_legume_and_fallow_rules_are_named_in_the_explanation() -> None:
    strategies = by_id(run(["maize", "fallow"]))
    assert "legume_after_cereal" in " ".join(strategies["maize_beans"]["why"])
    assert "bare_fallow_erosion_penalty" in " ".join(strategies["current"]["why"])
    assert any("bare" in r["description"].lower() for r in strategies["current"]["risks"])


def test_priorities_change_the_ranking() -> None:
    zero = dict.fromkeys(PRIORITIES, 0.0)

    def best(weights: dict[str, float]) -> str:
        strategies = run(["maize", "fallow"], weights=weights)
        return str(max(strategies, key=lambda s: s["overall_score"])["id"])

    assert best(zero | {"erosion_reduction": 1.0}) == "maize_cover_crop"
    assert "beans" in best(zero | {"nitrogen_fixing": 1.0})
    assert best(zero | {"income": 1.0}) in {"maize_beans", "maize_beans_cover_crop"}


def test_water_limited_constraint_changes_overall_scores() -> None:
    plain = by_id(run(["maize", "fallow"], weights={"water_use": 0.5}))
    limited = by_id(
        run(["maize", "fallow"], weights={"water_use": 0.5}, constraints={"water_limited": True})
    )
    assert plain["current"]["overall_score"] != limited["current"]["overall_score"]
    assert "water_limited" in " ".join(limited["current"]["why"])


def test_avoid_crops_penalises_but_never_drops_a_strategy() -> None:
    plain = by_id(run(["maize", "fallow"]))
    avoid = by_id(run(["maize", "fallow"], constraints={"avoid_crops": ["black_beans"]}))
    assert set(plain) == set(avoid)
    assert avoid["maize_beans"]["overall_score"] < plain["maize_beans"]["overall_score"]
    assert any("black_beans" in r["description"] for r in avoid["maize_beans"]["risks"])
    assert (
        avoid["maize_fallow" if "maize_fallow" in avoid else "current"]["overall_score"]
        == (plain["current"]["overall_score"])
    )


def test_unsupported_constraints_become_a_gap() -> None:
    for strategy in run(["maize", "fallow"], constraints={"budget": 100}):
        assert any("budget" in gap for gap in strategy["gaps"])


def test_unknown_crops_keep_neutral_scores_and_say_so() -> None:
    current = run(["maize", "quinoa"])[0]
    assert set(current["scores"].values()) == {0.5}
    assert any("quinoa" in gap and "neutral" in gap for gap in current["gaps"])
    assert current["needs_local_review"] is True
    assert current["risks"]
    assert len(run(["maize", "quinoa"])) >= 3


def test_empty_rotation_is_a_gap_not_a_crash() -> None:
    current = run([])[0]
    assert any("No current rotation" in gap for gap in current["gaps"])
    assert set(current["scores"].values()) == {0.5}


def test_same_input_same_output() -> None:
    assert run(["maize", "fallow"], DRY) == run(["maize", "fallow"], DRY)


def test_provenance_names_the_engine_and_the_traits_version() -> None:
    meta = engine.provenance()
    assert "rule engine" in meta["model_name"]
    assert "not a trained model" in meta["model_name"]
    assert meta["model_version"] == engine.ENGINE_VERSION
    assert meta["data_versions"]["crop_traits"] == "0.1.0-provisional"


@pytest.mark.parametrize("rotation", [["maize", "fallow"], ["unknown"], []])
def test_current_rotation_does_not_promise_equipment_savings(rotation: list[str]) -> None:
    current = run(rotation)[0]
    assert not any("no new equipment" in text.lower() for text in current["benefits"])
    assert any("equipment" in text.lower() for text in current["gaps"])


def test_each_enso_and_avoidance_adjustment_has_an_explanation_and_evidence() -> None:
    strategies = run(["maize", "fallow"], DRY, constraints={"avoid_crops": ["black_beans"]})
    for strategy in strategies:
        sources = {entry["source"] for entry in strategy["evidence"]}
        assert "rule_constant:enso_rainfall_equivalent" in sources
        if "black_beans" in strategy["sequence"]:
            assert "avoid_crops" in " ".join(strategy["why"])
            assert "rule_constant:avoid_crops" in sources


def test_cover_species_and_heuristic_range_are_disclosed() -> None:
    for strategy in run(["maize", "fallow"]):
        gaps = " ".join(strategy["gaps"]).lower()
        assert "not a calibrated probability" in gaps
        if "cover_crop" in strategy["sequence"]:
            assert "species" in gaps and "nitrogen" in gaps


@pytest.mark.parametrize("rain", [-80.0, -30.0, 0.0, 50.0, 100.0])
@pytest.mark.parametrize("temperature", [-5.0, 0.0, 8.0])
@pytest.mark.parametrize("phase", ["neutral", "el_nino", "la_nina"])
def test_scores_and_intervals_remain_finite_and_bounded_at_contract_limits(
    rain: float, temperature: float, phase: str
) -> None:
    import math

    scenario = {"rainfall_change_pct": rain, "temperature_change_c": temperature, "enso_phase": phase}
    for strategy in run(["maize", "fallow"], scenario):
        values = [strategy["overall_score"], *strategy["scores"].values()]
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
        interval = strategy["uncertainty"]
        assert 0 <= interval["low"] <= strategy["overall_score"] <= interval["high"] <= 1
