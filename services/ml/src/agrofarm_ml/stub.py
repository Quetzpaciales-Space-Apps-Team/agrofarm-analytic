"""Deterministic stub answers.

Fixed rules, not a trained model: the same input always gives the same output, so core-api
and the frontend can integrate before the real model and rule engine exist. The crops,
strategies and scores are placeholders until the agronomy decisions (AGR-001) land.
"""

from datetime import UTC, datetime
from typing import Any

from agrofarm_ml.schemas import RotationScoreRequest, Scenario, SimulateRequest

MODEL_NAME = "agrofarm-stub"
MODEL_VERSION = "stub-0.1.0"
TARGET = "yield_anomaly_pct"
UNIT = "%"
DISCLAIMER = "This tool supports decisions. It does not replace farmers or local agronomists."
MODELS = [
    {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "target": TARGET,
        "unit": UNIT,
        "status": "stub",
        "notes": "Fixed rules, not a trained model. Do not read the numbers as predictions.",
    }
]

BASELINE = {"p10": -12.0, "p50": 0.0, "p90": 11.0}
RAINFALL_EFFECT = 0.45  # yield % per rainfall %
TEMPERATURE_EFFECT = -2.5  # yield % per degree C
PHASE_EFFECT = {"el_nino": -4.0, "la_nina": 3.0, "neutral": 0.0}
OUT_OF_RANGE_RAINFALL_PCT = 60.0
OUT_OF_RANGE_TEMPERATURE_C = 4.0

PRIORITIES = (
    "water_use",
    "income",
    "soil_structure",
    "forage_value",
    "nitrogen_fixing",
    "erosion_reduction",
    "market_suitability",
)
DEFAULT_WEIGHT = 0.5
SCORE_HALF_RANGE = 0.1
PLACEHOLDER_GAP = "Scores are placeholders until the crop traits table is reviewed."


def _scores(*values: float) -> dict[str, float]:
    return dict(zip(PRIORITIES, values, strict=True))


_CATALOG: list[dict[str, Any]] = [
    {
        "id": "maize_fallow",
        "name": "Maize, then fallow",
        "sequence": ["maize", "fallow"],
        "scores": _scores(0.5, 0.6, 0.3, 0.1, 0.1, 0.3, 0.7),
        "why": ["One crop per year with the field resting afterwards."],
        "evidence": [
            {
                "statement": "About 93% of yearly rain falls from May to October "
                "at the reference station.",
                "source": "insivumeh_daily",
            }
        ],
        "benefits": ["Known practice, no new equipment."],
        "gaps": ["No yield records for this farm."],
        "risks": ["Bare soil from December to April raises erosion risk at the first heavy rains."],
    },
    {
        "id": "maize_beans",
        "name": "Maize, then black beans",
        "sequence": ["maize", "black_beans"],
        "scores": _scores(0.6, 0.7, 0.6, 0.2, 0.9, 0.6, 0.7),
        "why": [
            "Beans fix nitrogen for the next maize crop.",
            "A second crop shortens the bare-field period.",
        ],
        "evidence": [
            {"statement": "Legumes fix atmospheric nitrogen.", "source": "crop_traits_placeholder"}
        ],
        "benefits": ["Less nitrogen fertilizer needed.", "Soil covered for more months."],
        "gaps": ["Late-season rainfall is uncertain.", "No local bean yield data."],
        "risks": ["Beans sown late may meet the dry season before filling pods."],
    },
    {
        "id": "maize_cover_crop",
        "name": "Maize, then a cover crop",
        "sequence": ["maize", "cover_crop"],
        "scores": _scores(0.7, 0.4, 0.8, 0.6, 0.6, 0.9, 0.3),
        "why": ["Keeps the soil covered in the dry season."],
        "evidence": [
            {
                "statement": "90 days with 50 mm of rain or more were recorded "
                "from 1995 to 2025 at the reference station.",
                "source": "insivumeh_daily",
            }
        ],
        "benefits": ["Lower erosion risk.", "Better soil structure over several seasons."],
        "gaps": ["Seed availability and cost were not checked."],
        "risks": ["A cover crop gives no direct cash income."],
    },
]


def _provenance() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "data_versions": {},
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _contributions(scenario: Scenario) -> dict[str, float]:
    return {
        "rainfall_change_pct": round(scenario.rainfall_change_pct * RAINFALL_EFFECT, 2),
        "enso_phase": PHASE_EFFECT[scenario.enso_phase],
        "temperature_change_c": round(scenario.temperature_change_c * TEMPERATURE_EFFECT, 2),
    }


def simulate(request: SimulateRequest) -> dict[str, Any]:
    contributions = _contributions(request.scenario)
    shift = sum(contributions.values())
    out_of_range = []
    if abs(request.scenario.rainfall_change_pct) > OUT_OF_RANGE_RAINFALL_PCT:
        out_of_range.append("rainfall_change_pct")
    if abs(request.scenario.temperature_change_c) > OUT_OF_RANGE_TEMPERATURE_C:
        out_of_range.append("temperature_change_c")
    return {
        "crop": request.crop,
        "target": TARGET,
        "unit": UNIT,
        "baseline": dict(BASELINE),
        "scenario": {key: round(value + shift, 2) for key, value in BASELINE.items()},
        "attributions": [
            {"feature": feature, "contribution": value}
            for feature, value in sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
        ],
        "out_of_range_inputs": out_of_range,
        "provenance": _provenance(),
    }


def _overall(scores: dict[str, float], weights: dict[str, float]) -> float:
    used = {name: weights.get(name, DEFAULT_WEIGHT) for name in PRIORITIES}
    if sum(used.values()) == 0:
        used = dict.fromkeys(PRIORITIES, 1.0)
    return sum(scores[name] * used[name] for name in PRIORITIES) / sum(used.values())


def _strategy(entry: dict[str, Any], weights: dict[str, float], *, current: bool) -> dict[str, Any]:
    overall = _overall(entry["scores"], weights)
    why = list(entry["why"])
    if current:
        why.insert(0, "This is what the farm does today. It is the baseline for the comparison.")
    return {
        "id": "current" if current else entry["id"],
        "name": entry["name"],
        "is_current": current,
        "sequence": list(entry["sequence"]),
        "overall_score": round(overall, 2),
        "scores": dict(entry["scores"]),
        "uncertainty": {
            "low": round(max(0.0, overall - SCORE_HALF_RANGE), 2),
            "high": round(min(1.0, overall + SCORE_HALF_RANGE), 2),
        },
        "why": why,
        "evidence": [dict(item) for item in entry["evidence"]],
        "benefits": list(entry["benefits"]),
        "gaps": [*entry["gaps"], PLACEHOLDER_GAP],
        "risks": [{"description": text, "needs_local_review": True} for text in entry["risks"]],
        "needs_local_review": True,
    }


def _unknown_current(rotation: list[str]) -> dict[str, Any]:
    gap = (
        "This rotation is not in the crop table yet, so every score is a neutral 0.5."
        if rotation
        else "No current rotation was provided for this farm, so every score is a neutral 0.5."
    )
    return {
        "id": "current",
        "name": ", then ".join(rotation) if rotation else "Current rotation (not provided)",
        "sequence": rotation or ["unknown"],
        "scores": _scores(*[0.5] * len(PRIORITIES)),
        "why": [],
        "evidence": [{"statement": "Rotation as reported by the farm.", "source": "farm_profile"}],
        "benefits": ["Known practice, no new equipment."],
        "gaps": [gap],
        "risks": ["It cannot be compared fairly with the alternatives until it is described."],
    }


def score_rotations(request: RotationScoreRequest) -> dict[str, Any]:
    weights = request.priorities.weights
    rotation = list(request.current_rotation)
    known = next((entry for entry in _CATALOG if entry["sequence"] == rotation), None)
    strategies = [_strategy(known or _unknown_current(rotation), weights, current=True)]
    strategies += [
        _strategy(entry, weights, current=False) for entry in _CATALOG if entry is not known
    ]
    if known is not None:  # two alternatives left: the contract asks for at least three rows
        strategies.append(_strategy(_EXTRA, weights, current=False))
    return {"disclaimer": DISCLAIMER, "strategies": strategies, "provenance": _provenance()}


_EXTRA: dict[str, Any] = {
    "id": "maize_beans_cover_crop",
    "name": "Maize, then black beans, then a cover crop",
    "sequence": ["maize", "black_beans", "cover_crop"],
    "scores": _scores(0.6, 0.6, 0.8, 0.5, 0.9, 0.9, 0.5),
    "why": ["Combines nitrogen fixing with soil cover in the dry season."],
    "evidence": [
        {"statement": "Legumes fix atmospheric nitrogen.", "source": "crop_traits_placeholder"}
    ],
    "benefits": ["Soil covered most of the year."],
    "gaps": ["Whether three crops fit in the local season was not checked."],
    "risks": ["A longer sequence needs more labor and water late in the year."],
}
