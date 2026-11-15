"""ML-040: every rule is a named pure function and is tested one by one."""

import pytest

from agrofarm_ml.rotations import rules
from agrofarm_ml.rotations.traits import Crop, load_traits


def crops(*names: str) -> list[Crop]:
    found = [load_traits().resolve(name) for name in names]
    assert all(found)
    return [crop for crop in found if crop]


def test_base_scores_are_the_mean_of_the_crop_traits_and_water_use_is_a_fit() -> None:
    scores = rules.base_scores(crops("maize", "fallow"))
    assert scores["income"] == pytest.approx(0.3)  # mean of 0.6 and 0.0
    assert scores["water_use"] == pytest.approx(0.65)  # 1 - mean water demand (0.7, 0.0)
    assert set(scores) == set(rules.PRIORITIES)


def test_legume_after_cereal_raises_nitrogen_and_a_little_soil_structure() -> None:
    adjustments = rules.legume_after_cereal(crops("maize", "black_beans"))
    assert {a.priority: a.delta for a in adjustments} == {
        "nitrogen_fixing": rules.LEGUME_AFTER_CEREAL_NITROGEN_BONUS,
        "soil_structure": rules.LEGUME_AFTER_CEREAL_SOIL_BONUS,
    }
    assert all(a.rule == "legume_after_cereal" for a in adjustments)


def test_legume_rule_does_not_fire_without_a_cereal_before_a_legume() -> None:
    assert rules.legume_after_cereal(crops("maize", "fallow")) == []
    assert rules.legume_after_cereal(crops("black_beans", "cover_crop")) == []
    assert rules.legume_after_cereal(crops("black_beans")) == []


def test_the_rotation_is_a_cycle_so_a_legume_first_still_follows_the_cereal() -> None:
    assert rules.legume_after_cereal(crops("black_beans", "maize"))


def test_bare_fallow_penalty_scales_with_the_share_of_fallow() -> None:
    half = rules.bare_fallow_erosion_penalty(crops("maize", "fallow"))
    assert [a.delta for a in half] == [pytest.approx(-0.5 * rules.BARE_FALLOW_EROSION_PENALTY)]
    assert half[0].priority == "erosion_reduction"
    assert rules.bare_fallow_erosion_penalty(crops("maize", "black_beans")) == []


@pytest.mark.parametrize(
    ("phase", "rain", "expected"),
    [("neutral", 0.0, 0.0), ("el_nino", -20.0, -30.0), ("la_nina", 5.0, 15.0)],
)
def test_effective_rainfall_adds_the_enso_equivalent(
    phase: str, rain: float, expected: float
) -> None:
    assert rules.effective_rainfall_change_pct(rain, phase) == pytest.approx(expected)


def test_dry_scenario_lowers_water_fit_more_for_water_demanding_sequences() -> None:
    thirsty = rules.dry_scenario_water_fit(crops("maize", "black_beans"), -30.0)
    light = rules.dry_scenario_water_fit(crops("maize", "fallow"), -30.0)
    assert thirsty[0].delta < light[0].delta < 0
    assert thirsty[0].rule == "dry_scenario_water_fit"
    assert thirsty[0].priority == "water_use"


def test_dry_rule_is_silent_when_it_is_not_drier() -> None:
    assert rules.dry_scenario_water_fit(crops("maize"), 0.0) == []
    assert rules.dry_scenario_water_fit(crops("maize"), 20.0) == []


def test_wet_scenario_hurts_erosion_most_where_the_soil_is_least_covered() -> None:
    bare = rules.wet_scenario_erosion_exposure(crops("maize", "fallow"), 40.0)
    covered = rules.wet_scenario_erosion_exposure(crops("maize", "cover_crop"), 40.0)
    assert bare[0].delta < covered[0].delta < 0
    assert rules.wet_scenario_erosion_exposure(crops("maize"), -10.0) == []


def test_warming_raises_water_demand_only_when_it_is_warmer() -> None:
    assert rules.warming_water_fit(crops("maize"), 2.0)[0].delta < 0
    assert rules.warming_water_fit(crops("maize"), 0.0) == []
    assert rules.warming_water_fit(crops("maize"), -2.0) == []


def test_every_adjustment_is_small() -> None:
    """The scenario is modest by design: one rule never moves a score by more than 0.25."""
    extreme = [
        *rules.dry_scenario_water_fit(crops("maize", "black_beans"), -80.0),
        *rules.wet_scenario_erosion_exposure(crops("fallow"), 100.0),
        *rules.warming_water_fit(crops("maize", "black_beans"), 8.0),
    ]
    assert all(abs(a.delta) <= 0.25 for a in extreme)


def test_apply_clips_to_zero_one_and_keeps_the_trace() -> None:
    base = dict.fromkeys(rules.PRIORITIES, 0.95)
    adjusted = rules.apply_adjustments(
        base,
        [rules.Adjustment("r", "income", 0.3, "x"), rules.Adjustment("r", "water_use", -2, "y")],
    )
    assert adjusted["income"] == 1.0
    assert adjusted["water_use"] == 0.0
    assert base["income"] == 0.95  # input untouched


def test_weights_default_to_half_and_all_zero_means_equal() -> None:
    weights = rules.effective_weights({}, {})
    assert set(weights.values()) == {rules.DEFAULT_WEIGHT}
    zero = rules.effective_weights(dict.fromkeys(rules.PRIORITIES, 0.0), {})
    assert set(zero.values()) == {1.0}


def test_water_limited_constraint_raises_the_water_weight() -> None:
    plain = rules.effective_weights({"water_use": 0.4}, {})
    limited = rules.effective_weights({"water_use": 0.4}, {"water_limited": True})
    assert limited["water_use"] == pytest.approx(
        plain["water_use"] * rules.WATER_LIMITED_MULTIPLIER
    )
    assert limited["income"] == plain["income"]


def test_overall_score_is_the_weighted_mean() -> None:
    scores = dict.fromkeys(rules.PRIORITIES, 0.0) | {"income": 1.0}
    only_income = dict.fromkeys(rules.PRIORITIES, 0.0) | {"income": 1.0}
    assert rules.overall_score(scores, only_income) == pytest.approx(1.0)
    assert rules.overall_score(scores, dict.fromkeys(rules.PRIORITIES, 1.0)) == pytest.approx(1 / 7)


def test_avoid_crops_penalises_a_strategy_that_contains_one() -> None:
    hit = rules.avoided_crops(crops("maize", "fallow"), {"avoid_crops": ["Maize", "pumpkin"]})
    assert hit == ["maize"]
    assert rules.avoided_crops(crops("maize"), {}) == []
    assert rules.avoided_crops(crops("maize"), {"avoid_crops": "maize"}) == []  # not a list


def test_unsupported_constraints_are_listed() -> None:
    assert rules.unsupported_constraints({"avoid_crops": [], "budget": 3}) == ["budget"]


def test_uncertainty_grows_with_a_stronger_scenario_and_stays_in_bounds() -> None:
    calm = rules.uncertainty_interval(0.5, 0.0, 0.0)
    strong = rules.uncertainty_interval(0.5, -60.0, 3.0)
    assert strong[1] - strong[0] > calm[1] - calm[0]
    low, high = rules.uncertainty_interval(0.99, -80.0, 8.0)
    assert 0.0 <= low <= 0.99 <= high <= 1.0
