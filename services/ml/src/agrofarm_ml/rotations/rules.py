"""Rotation rules: named pure functions, no I/O.

Every constant below is a PROVISIONAL assumption chosen by the AI assistant, not an agronomic
finding. It needs review by a local agronomist (AGR-001). Each rule returns `Adjustment`
records so the response can say which rule moved which score and by how much.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agrofarm_ml.rotations.traits import PRIORITIES, Crop

# PROVISIONAL rule constants (see docs of each rule).
LEGUME_AFTER_CEREAL_NITROGEN_BONUS = 0.25  # added to nitrogen_fixing when a legume follows a cereal
LEGUME_AFTER_CEREAL_SOIL_BONUS = 0.05  # added to soil_structure in the same case
BARE_FALLOW_EROSION_PENALTY = 0.15  # subtracted from erosion_reduction, times the fallow share
DRY_WATER_FIT_SLOPE = 0.5  # water_use fit lost per 100% rainfall drop, times mean water demand
WET_EROSION_SLOPE = 0.2  # erosion_reduction lost per 100% rainfall rise, times bare-soil share
WARMING_WATER_SLOPE = 0.03  # water_use fit lost per degree C, times mean water demand
ENSO_RAINFALL_EQUIVALENT_PCT = {"el_nino": -10.0, "la_nina": 10.0, "neutral": 0.0}
DEFAULT_WEIGHT = 0.5  # weight of a priority the farmer did not set
WATER_LIMITED_MULTIPLIER = 2.0  # water_use weight factor when the constraint water_limited is set
AVOIDED_CROP_PENALTY = 0.15  # subtracted from the overall score for a crop the farmer avoids
BASE_HALF_RANGE = 0.10  # uncertainty half width with no scenario
RAINFALL_HALF_RANGE = 0.10  # extra half width at a 100% effective rainfall change
TEMPERATURE_HALF_RANGE = 0.02  # extra half width per degree C of warming or cooling
MAX_HALF_RANGE = 0.40
SUPPORTED_CONSTRAINTS = ("avoid_crops", "water_limited")

__all__ = ["PRIORITIES", "Adjustment"]


@dataclass(frozen=True)
class Adjustment:
    rule: str  # the rule name shown to the reader
    priority: str
    delta: float
    reason: str


def _mean(crops: Sequence[Crop], trait: str) -> float:
    return sum(crop.trait(trait) for crop in crops) / len(crops)


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def base_scores(crops: Sequence[Crop]) -> dict[str, float]:
    """Mean of the crop traits over the sequence. water_use is a FIT: 1 minus mean water demand."""
    scores = {name: _mean(crops, name) for name in PRIORITIES if name != "water_use"}
    scores["water_use"] = 1.0 - _mean(crops, "water_demand")
    return {name: scores[name] for name in PRIORITIES}


def legume_after_cereal(crops: Sequence[Crop]) -> list[Adjustment]:
    """A legume right after a cereal (the rotation repeats, so the last step precedes the first)
    leaves nitrogen for the next cereal. Bonus is flat: one pair is enough."""
    n = len(crops)
    pairs = [
        (crops[i], crops[(i + 1) % n])
        for i in range(n)
        if n > 1 and crops[i].group == "cereal" and crops[(i + 1) % n].group == "legume"
    ]
    if not pairs:
        return []
    cereal, legume = pairs[0]
    reason = f"{legume.crop_id} follows {cereal.crop_id}, so the legume may leave nitrogen behind"
    return [
        Adjustment(
            "legume_after_cereal", "nitrogen_fixing", LEGUME_AFTER_CEREAL_NITROGEN_BONUS, reason
        ),
        Adjustment("legume_after_cereal", "soil_structure", LEGUME_AFTER_CEREAL_SOIL_BONUS, reason),
    ]


def bare_fallow_erosion_penalty(crops: Sequence[Crop]) -> list[Adjustment]:
    share = sum(1 for crop in crops if crop.group == "fallow") / len(crops)
    if share == 0:
        return []
    return [
        Adjustment(
            "bare_fallow_erosion_penalty",
            "erosion_reduction",
            -BARE_FALLOW_EROSION_PENALTY * share,
            f"bare fallow is {share:.0%} of the sequence and leaves the soil exposed",
        )
    ]


def effective_rainfall_change_pct(rainfall_change_pct: float, enso_phase: str) -> float:
    """Rainfall change asked for plus a fixed equivalent for the ENSO phase (provisional)."""
    return rainfall_change_pct + ENSO_RAINFALL_EQUIVALENT_PCT[enso_phase]


def dry_scenario_water_fit(
    crops: Sequence[Crop], effective_rainfall_pct: float
) -> list[Adjustment]:
    """A drier scenario lowers the water_use fit of water-demanding sequences."""
    demand = _mean(crops, "water_demand")
    delta = -DRY_WATER_FIT_SLOPE * (max(0.0, -effective_rainfall_pct) / 100.0) * demand
    if delta == 0:
        return []
    return [
        Adjustment(
            "dry_scenario_water_fit",
            "water_use",
            delta,
            f"effective rainfall change is {effective_rainfall_pct:+.0f}% and the sequence "
            f"has a mean water demand of {demand:.2f}",
        )
    ]


def wet_scenario_erosion_exposure(
    crops: Sequence[Crop], effective_rainfall_pct: float
) -> list[Adjustment]:
    """A wetter scenario hurts erosion protection where the soil is least covered."""
    exposed = 1.0 - _mean(crops, "soil_cover")
    delta = -WET_EROSION_SLOPE * (max(0.0, effective_rainfall_pct) / 100.0) * exposed
    if delta == 0:
        return []
    return [
        Adjustment(
            "wet_scenario_erosion_exposure",
            "erosion_reduction",
            delta,
            f"effective rainfall change is {effective_rainfall_pct:+.0f}% and "
            f"{exposed:.0%} of the soil is uncovered on average",
        )
    ]


def warming_water_fit(crops: Sequence[Crop], temperature_change_c: float) -> list[Adjustment]:
    """Warming raises water demand, so it lowers the water_use fit."""
    delta = -WARMING_WATER_SLOPE * max(0.0, temperature_change_c) * _mean(crops, "water_demand")
    if delta == 0:
        return []
    return [
        Adjustment(
            "warming_water_fit",
            "water_use",
            delta,
            f"temperature is {temperature_change_c:+.1f} C and warmer weather raises water demand",
        )
    ]


def apply_adjustments(
    scores: Mapping[str, float], adjustments: Sequence[Adjustment]
) -> dict[str, float]:
    result = dict(scores)
    for item in adjustments:
        result[item.priority] = clip(result[item.priority] + item.delta)
    return result


def effective_weights(
    weights: Mapping[str, float], constraints: Mapping[str, Any]
) -> dict[str, float]:
    """Weights the farmer set (missing ones default to 0.5), changed by constraints.
    All zero weights fall back to equal weights."""
    used = {name: float(weights.get(name, DEFAULT_WEIGHT)) for name in PRIORITIES}
    if constraints.get("water_limited") is True:
        used["water_use"] *= WATER_LIMITED_MULTIPLIER
    if sum(used.values()) == 0:
        used = dict.fromkeys(PRIORITIES, 1.0)
    return used


def overall_score(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(scores[name] * weights[name] for name in PRIORITIES) / sum(weights.values())


def avoided_crops(crops: Sequence[Crop], constraints: Mapping[str, Any]) -> list[str]:
    asked = constraints.get("avoid_crops")
    if not isinstance(asked, list):
        return []
    wanted = {str(item).strip().lower().replace(" ", "_") for item in asked}
    return [crop.crop_id for crop in crops if crop.crop_id in wanted]


def unsupported_constraints(constraints: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in constraints if key not in SUPPORTED_CONSTRAINTS)


def uncertainty_interval(
    overall: float,
    effective_rainfall_pct: float,
    temperature_change_c: float,
    *,
    known: bool = True,
) -> tuple[float, float]:
    """Wider when the scenario is stronger and when the rotation is not in the traits table."""
    if not known:
        half = 0.25
    else:
        half = (
            BASE_HALF_RANGE
            + RAINFALL_HALF_RANGE * min(1.0, abs(effective_rainfall_pct) / 100.0)
            + TEMPERATURE_HALF_RANGE * abs(temperature_change_c)
        )
    half = min(half, MAX_HALF_RANGE)
    return clip(overall - half), clip(overall + half)
