"""Deterministic stub answers.

Fixed rules, not a trained model: the same input always gives the same output. This
module answers /v1/simulate only; rotation scoring moved to agrofarm_ml.rotations.engine.
"""

from datetime import UTC, datetime
from typing import Any

from agrofarm_ml.schemas import Scenario, SimulateRequest

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


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _provenance() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "data_versions": {},
        "generated_at": now_iso(),
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
