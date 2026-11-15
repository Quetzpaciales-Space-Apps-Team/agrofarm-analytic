"""Rotation engine: scores the current rotation and candidate sequences with the rules.

Pure functions, no I/O apart from reading the packaged traits table. Rule based, NOT a trained
model. Traits and rule constants are provisional and need review by a local agronomist.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from agrofarm_ml.rotations import rules
from agrofarm_ml.rotations.rules import PRIORITIES, Adjustment
from agrofarm_ml.rotations.traits import Crop, TraitsTable, load_traits

ENGINE_NAME = "agrofarm-rotation-rule-engine"
ENGINE_VERSION = "rules-0.1.1"
MODEL_NAME = f"{ENGINE_NAME} (rule engine, provisional crop traits, not a trained model)"
GOOD_FIT = 0.65  # a priority score at or above this is listed as a benefit
LOW_INCOME = 0.35
THIRSTY = 0.4  # mean water demand above this earns a drought risk in a drier scenario

# Candidate sequences compared with the current rotation (provisional, from the pilot crops).
CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("maize_fallow", ("maize", "fallow")),
    ("maize_beans", ("maize", "black_beans")),
    ("maize_cover_crop", ("maize", "cover_crop")),
    ("maize_beans_cover_crop", ("maize", "black_beans", "cover_crop")),
)
LOCAL_REVIEW = "needs review by a local agronomist"


def traits_version_label(table: TraitsTable) -> str:
    return f"{table.version}-{table.manifest['status']}"


def provenance(table: TraitsTable | None = None) -> dict[str, Any]:
    table = table or load_traits()
    return {
        "model_name": MODEL_NAME,
        "model_version": ENGINE_VERSION,
        "data_versions": {"crop_traits": traits_version_label(table)},
    }


def model_card(table: TraitsTable | None = None) -> dict[str, Any]:
    """The entry for GET /v1/models."""
    table = table or load_traits()
    return {
        "name": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "target": "rotation_score",
        "unit": "score_0_to_1",
        "status": "rule_engine",
        "notes": "Rule engine over a provisional crop traits table. Not a trained model. "
        f"Needs review by a local agronomist. Crop traits {traits_version_label(table)}.",
        "data_versions": {"crop_traits": traits_version_label(table)},
    }


def _name(crops: Sequence[Crop]) -> str:
    parts = [crop.display_name.lower() for crop in crops]
    return (
        ", then ".join(parts).capitalize()
        if len(parts) == 1
        else (parts[0].capitalize() + "".join(f", then {part}" for part in parts[1:]))
    )


def _line(item: Adjustment) -> str:
    return (
        f"Rule {item.rule}: {item.reason}; {item.priority} {item.delta:+.3f} "
        "(provisional rule constant)."
    )


def _evidence(crops: Sequence[Crop], table: TraitsTable) -> list[dict[str, str]]:
    return [
        {
            "statement": f"{crop.display_name}: water demand {crop.trait('water_demand'):.2f}, "
            f"income {crop.trait('income'):.2f}, "
            f"nitrogen fixing {crop.trait('nitrogen_fixing'):.2f}, "
            f"soil cover {crop.trait('soil_cover'):.2f} (provisional placeholder values).",
            "source": f"crop_traits:{table.version}:{crop.crop_id}",
        }
        for crop in crops
    ]


def _gaps(table: TraitsTable, constraints: Mapping[str, Any]) -> list[str]:
    gaps = [
        f"Crop traits table {table.version} retains {table.manifest['status']} source status: "
        f"{LOCAL_REVIEW} before applying it to this farm.",
        "No yield records, soil data or planting dates of this farm enter the scores.",
        "Equipment needs and costs have not been assessed for this farm.",
        "The score range is a provisional heuristic, not a calibrated probability of success.",
    ]
    if table.manifest.get("human_review_confirmation", {}).get("review_occurred") is True:
        gaps.append(
            "The project owner confirmed an agronomy review of the crop traits. "
            "The reviewer identity, reviewed revision and corrections are not recorded; "
            "source verification and rule-threshold review are not established."
        )
    ignored = rules.unsupported_constraints(constraints)
    if ignored:
        gaps.append(f"These constraints are not understood and were ignored: {', '.join(ignored)}.")
    return gaps


def _known_strategy(
    strategy_id: str,
    crops: list[Crop],
    *,
    current: bool,
    weights: Mapping[str, float],
    constraints: Mapping[str, Any],
    scenario: Mapping[str, Any],
    table: TraitsTable,
) -> dict[str, Any]:
    effective_rain = rules.effective_rainfall_change_pct(
        float(scenario["rainfall_change_pct"]), str(scenario["enso_phase"])
    )
    temperature = float(scenario["temperature_change_c"])
    adjustments = [
        *rules.legume_after_cereal(crops),
        *rules.bare_fallow_erosion_penalty(crops),
        *rules.dry_scenario_water_fit(crops, effective_rain),
        *rules.wet_scenario_erosion_exposure(crops, effective_rain),
        *rules.warming_water_fit(crops, temperature),
    ]
    scores = rules.apply_adjustments(rules.base_scores(crops), adjustments)
    used = rules.effective_weights(weights, constraints)
    overall = rules.overall_score(scores, used)
    hits = rules.avoided_crops(crops, constraints)
    if hits:
        overall = rules.clip(overall - rules.AVOIDED_CROP_PENALTY)
    low, high = rules.uncertainty_interval(overall, effective_rain, temperature)

    why = [_line(item) for item in adjustments]
    phase = str(scenario["enso_phase"])
    if phase != "neutral":
        shift = rules.ENSO_RAINFALL_EQUIVALENT_PCT[phase]
        why.append(
            f"Rule enso_rainfall_equivalent: {phase} counts as {shift:+.0f}% rainfall, so the "
            f"effective rainfall change is {effective_rain:+.0f}% (provisional rule constant)."
        )
    if constraints.get("water_limited") is True:
        why.append(
            "Constraint water_limited: the water_use weight is multiplied by "
            f"{rules.WATER_LIMITED_MULTIPLIER:g}."
        )
    if hits:
        why.append(
            f"Constraint avoid_crops: contains {', '.join(hits)}; the overall score is reduced "
            f"by {rules.AVOIDED_CROP_PENALTY:g} before clipping to 0..1."
        )
    if not why:
        why.append("No rule moved the base scores, which are the mean of the crop traits.")
    top = max(PRIORITIES, key=lambda name: scores[name] * used[name])
    why.append(f"Strongest contribution to the overall score: {top} ({scores[top]:.2f}).")
    if current:
        why.insert(0, "This is what the farm does today. It is the baseline for the comparison.")

    evidence = _evidence(crops, table)
    evidence.append(
        {
            "statement": f"Scenario: rainfall {scenario['rainfall_change_pct']:+g}%, "
            f"temperature {temperature:+g} C, ENSO {phase}.",
            "source": "scenario_request",
        }
    )
    evidence += [
        {
            "statement": f"Rule {item.rule} uses a provisional constant, not a measured value.",
            "source": f"rule_constant:{item.rule}",
        }
        for item in {i.rule: i for i in adjustments}.values()
    ]
    for name, active in (
        ("enso_rainfall_equivalent", phase != "neutral"),
        ("water_limited", constraints.get("water_limited") is True),
        ("avoid_crops", bool(hits)),
    ):
        if active:
            evidence.append(
                {
                    "statement": f"{name} uses a provisional rule, not a measured effect.",
                    "source": f"rule_constant:{name}",
                }
            )

    benefits = [
        f"Good fit for {name} (score {scores[name]:.2f}, provisional)."
        for name in PRIORITIES
        if scores[name] >= GOOD_FIT
    ]
    if current:
        benefits.insert(0, "Provides the supplied baseline for comparison.")
    if not benefits:
        benefits.append("No priority scores high on the current weights.")

    gaps = _gaps(table, constraints)
    if any(crop.group == "cover" for crop in crops):
        gaps.append(
            "The cover crop species is not selected; nitrogen fixation assumes a legume "
            "mixture and is not verified for this farm."
        )
    risks = _risks(crops, scores, effective_rain, hits)
    return {
        "id": "current" if current else strategy_id,
        "name": _name(crops),
        "is_current": current,
        "sequence": [crop.crop_id for crop in crops],
        "overall_score": round(overall, 3),
        "scores": {name: round(scores[name], 3) for name in PRIORITIES},
        "uncertainty": {"low": round(low, 3), "high": round(high, 3)},
        "why": why,
        "evidence": evidence,
        "benefits": benefits,
        "gaps": gaps,
        "risks": [{"description": text, "needs_local_review": True} for text in risks],
        "needs_local_review": True,
    }


def _risks(
    crops: Sequence[Crop], scores: Mapping[str, float], effective_rain: float, hits: list[str]
) -> list[str]:
    risks: list[str] = []
    if any(crop.group == "fallow" for crop in crops):
        risks.append("Bare soil in the fallow step raises erosion risk at the first heavy rains.")
    if effective_rain < 0 and sum(c.trait("water_demand") for c in crops) / len(crops) > THIRSTY:
        risks.append("Water-demanding crops may run short of water in a drier scenario.")
    if any(crop.group == "cover" for crop in crops):
        risks.append("A cover crop gives little direct cash income.")
    elif scores["income"] < LOW_INCOME:
        risks.append("Cash income is low on this sequence.")
    risks += [f"Contains {crop}, which the farmer asked to avoid." for crop in hits]
    risks.append(f"Scores rest on provisional crop traits: {LOCAL_REVIEW}.")
    return risks


def _unknown_strategy(
    rotation: Sequence[str],
    unknown: Sequence[str],
    table: TraitsTable,
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    if rotation:
        gap = (
            f"Crops not in the crop traits table: {', '.join(unknown)}. "
            "Every score is a neutral 0.5."
        )
    else:
        gap = "No current rotation was provided for this farm, so every score is a neutral 0.5."
    low, high = rules.uncertainty_interval(0.5, 0.0, 0.0, known=False)
    return {
        "id": "current",
        "name": ", then ".join(rotation) if rotation else "Current rotation (not provided)",
        "is_current": True,
        "sequence": list(rotation) or ["unknown"],
        "overall_score": 0.5,
        "scores": dict.fromkeys(PRIORITIES, 0.5),
        "uncertainty": {"low": low, "high": high},
        "why": [
            "This is what the farm does today. It is the baseline for the comparison.",
            "Rule neutral_for_unknown_crops: nothing is known about this rotation, so no "
            "priority is scored above or below 0.5.",
        ],
        "evidence": [{"statement": "Rotation as reported by the farm.", "source": "farm_profile"}],
        "benefits": ["Provides the supplied baseline for comparison."],
        "gaps": [gap, *_gaps(table, constraints)],
        "risks": [
            {
                "description": "It cannot be compared fairly with the alternatives until it "
                "is described with crops from the traits table.",
                "needs_local_review": True,
            }
        ],
        "needs_local_review": True,
    }


def score(
    current_rotation: Sequence[str],
    weights: Mapping[str, float],
    constraints: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The current rotation first, then the candidates. Always at least three strategies."""
    table = load_traits()
    resolved = [table.resolve(name) for name in current_rotation]
    known = [crop for crop in resolved if crop is not None]
    if current_rotation and len(known) == len(resolved):
        sequence = tuple(crop.crop_id for crop in known)
        current = _known_strategy(
            "current",
            known,
            current=True,
            weights=weights,
            constraints=constraints,
            scenario=scenario,
            table=table,
        )
        matched = next((cid for cid, seq in CANDIDATES if seq == sequence), None)
    else:
        unknown = [n for n, crop in zip(current_rotation, resolved, strict=True) if crop is None]
        current = _unknown_strategy(current_rotation, unknown, table, constraints)
        matched = None
    strategies = [current]
    for candidate_id, names in CANDIDATES:
        if candidate_id == matched:
            continue
        crops = [crop for name in names if (crop := table.resolve(name)) is not None]
        strategies.append(
            _known_strategy(
                candidate_id,
                crops,
                current=False,
                weights=weights,
                constraints=constraints,
                scenario=scenario,
                table=table,
            )
        )
    return strategies
