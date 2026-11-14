"""Scenario function: what does the model expect for a month under an ENSO state?"""

from typing import Any

import pandas as pd

from agrofarm_ml.features import ENSO_LAGS, FEATURES, month_cycle
from agrofarm_ml.model import MODEL_VERSION, RainfallModel

# RONI value used for a named phase. One degree is a moderate event.
PHASE_RONI = {"el_nino": 1.0, "neutral": 0.0, "la_nina": -1.0}


def _features(month_of_year: int, roni: float) -> pd.DataFrame:
    sin, cos = month_cycle(month_of_year)
    row = {"month_sin": float(sin), "month_cos": float(cos)}
    row.update({f"roni_lag{lag}": roni for lag in ENSO_LAGS})
    return pd.DataFrame([row])[FEATURES]


def _range(model: RainfallModel, month_of_year: int, roni: float) -> dict[str, float]:
    predicted = model.predict(_features(month_of_year, roni)).iloc[0]
    return {name: round(float(predicted[name]), 1) for name in ("p10", "p50", "p90")}


def simulate(
    model: RainfallModel,
    month_of_year: int,
    enso_phase: str | None = None,
    roni: float | None = None,
    rainfall_change_pct: float = 0.0,
) -> dict[str, Any]:
    """Baseline (neutral ENSO) against a scenario, each with a p10 to p90 range."""
    if not 1 <= month_of_year <= 12:
        raise ValueError("month_of_year must be between 1 and 12")
    if (enso_phase is None) == (roni is None):
        raise ValueError("Give exactly one of enso_phase or roni")
    if enso_phase is not None:
        if enso_phase not in PHASE_RONI:
            raise ValueError(f"enso_phase must be one of {sorted(PHASE_RONI)}")
        roni = PHASE_RONI[enso_phase]
    assert roni is not None  # noqa: S101

    low, high = model.feature_ranges["roni_lag0"]
    out_of_range = [] if low <= roni <= high else ["roni"]
    assumptions = [
        "The ENSO index is held at the same value for the month and the six months before it."
    ]
    scenario = _range(model, month_of_year, roni)
    if rainfall_change_pct:
        factor = 1.0 + rainfall_change_pct / 100.0
        scenario = {name: round(value * factor, 1) for name, value in scenario.items()}
        assumptions.append(
            f"rainfall_change_pct={rainfall_change_pct:g} is applied on top of the model output. "
            "It is a what-if chosen by the user, not a prediction."
        )
    return {
        "target": "monthly_rainfall",
        "unit": "mm per month",
        "month_of_year": month_of_year,
        "roni": roni,
        "baseline": _range(model, month_of_year, PHASE_RONI["neutral"]),
        "scenario": scenario,
        "out_of_range_inputs": out_of_range,
        "assumptions": assumptions,
        "model_version": MODEL_VERSION,
    }
