"""Gradient boosting quantile model (p10, p50, p90) of the seasonal peak NDVI anomaly.

Target: peak NDVI of the cropland of a cell in the rainy season, as a percentage of that cell's
normal. The benchmark is climatology: the same quantiles of the training anomalies for every
row, which is what "expect a normal year" means.

Rows of the same year share the weather, so they are not independent. Validation therefore
leaves out one whole year at a time, and the verdict looks at years, not only at rows.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agrofarm_ml.model import pinball_loss
from agrofarm_ml.ndvi.table import TARGET, with_anomalies

__all__ = ["FEATURES", "TARGET", "NdviModel", "leave_one_year_out", "train_model", "verdict"]

MODEL_VERSION = "ndvi-peak-experimental-0.1.0"
QUANTILES = (0.1, 0.5, 0.9)
NAMES = ["p10", "p50", "p90"]
FEATURES = [
    "rain_anom_pct",
    "rain_early_anom_pct",
    "rain_late_anom_pct",
    "temp_anom_c",
    "roni_season",
    "roni_pre_season",
    "cropland_fraction",
    "ndvi_normal",
    "rain_normal_mm",
    "temp_normal_c",
]
PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_data_in_leaf": 200,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
ROUNDS = 150

# Written before any model was fit. Changing these after seeing results is not allowed.
MIN_POOLED_SKILL = 0.05
MIN_SHARE_YEARS_POSITIVE = 0.6
RULE = (
    "Leave one whole year out. The model works only if the pooled pinball loss is at least "
    "5 % lower than climatology, the skill is positive in at least 60 % of the years, and it "
    "is positive in at least one El Nino year and one La Nina year."
)


@dataclass
class NdviModel:
    boosters: dict[str, lgb.Booster]
    feature_ranges: dict[str, tuple[float, float]]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = features[FEATURES].to_numpy(dtype=float)
        raw = np.column_stack([self.boosters[name].predict(matrix) for name in NAMES])
        # Independent quantile models can cross. Sorting restores p10 <= p50 <= p90.
        return pd.DataFrame(np.sort(raw, axis=1), columns=NAMES, index=features.index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, booster in self.boosters.items():
            booster.save_model(str(directory / f"{name}.lgb.txt"))
        (directory / "feature_ranges.json").write_text(
            json.dumps(self.feature_ranges, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "NdviModel":
        boosters = {
            name: lgb.Booster(model_file=str(directory / f"{name}.lgb.txt")) for name in NAMES
        }
        ranges = json.loads((directory / "feature_ranges.json").read_text(encoding="utf-8"))
        return cls(boosters, {key: (low, high) for key, (low, high) in ranges.items()})


def train_model(train: pd.DataFrame, seed: int = 0) -> NdviModel:
    matrix = train[FEATURES].to_numpy(dtype=float)
    target = train[TARGET].to_numpy(dtype=float)
    boosters = {}
    for name, quantile in zip(NAMES, QUANTILES, strict=True):
        params = {**PARAMS, "alpha": quantile, "seed": seed}
        boosters[name] = lgb.train(params, lgb.Dataset(matrix, label=target), ROUNDS)
    ranges = {
        column: (float(train[column].min()), float(train[column].max())) for column in FEATURES
    }
    return NdviModel(boosters, ranges)


def _mean_pinball(actual: np.ndarray, predicted: np.ndarray) -> float:
    losses = [
        pinball_loss(actual, predicted[:, index], quantile)
        for index, quantile in enumerate(QUANTILES)
    ]
    return float(np.mean(losses))


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return round(float(np.corrcoef(first, second)[0, 1]), 3)


def leave_one_year_out(
    table: pd.DataFrame,
    phases: "pd.Series[str]",
    seed: int = 0,
    on_fold: Callable[[list[int], list[int]], None] | None = None,
) -> dict[str, Any]:
    """Predict every year with a model, and with cell normals, that never saw that year."""
    per_year: list[dict[str, Any]] = []
    model_loss = benchmark_loss = 0.0
    for year in sorted(int(value) for value in table["year"].unique()):
        train_rows, test_rows = table[table["year"] != year], table[table["year"] == year]
        train = with_anomalies(train_rows, train_rows)
        test = with_anomalies(test_rows, train_rows)
        if test.empty:
            continue
        if on_fold is not None:
            on_fold(sorted(set(train["year"])), sorted(set(test["year"])))
        actual = test[TARGET].to_numpy(dtype=float)
        predicted = train_model(train, seed=seed).predict(test[FEATURES]).to_numpy()
        normal_year = np.quantile(train[TARGET].to_numpy(dtype=float), QUANTILES)
        benchmark = np.tile(normal_year, (len(actual), 1))
        loss, reference = _mean_pinball(actual, predicted), _mean_pinball(actual, benchmark)
        model_loss += loss * len(actual)
        benchmark_loss += reference * len(actual)
        inside = (actual >= predicted[:, 0]) & (actual <= predicted[:, 2])
        per_year.append(
            {
                "year": year,
                "phase": str(phases.get(year, "unknown")),
                "rows": int(len(actual)),
                "skill": round(1.0 - loss / reference, 3),
                "mean_actual_pct": round(float(actual.mean()), 2),
                "mean_p50_pct": round(float(predicted[:, 1].mean()), 2),
                "coverage_p10_p90": round(float(inside.mean()), 3),
            }
        )
    positive = [fold for fold in per_year if fold["skill"] > 0]
    rows = sum(fold["rows"] for fold in per_year)
    return {
        "pooled_skill": round(1.0 - model_loss / benchmark_loss, 4),
        "years": len(per_year),
        "rows": rows,
        "share_years_positive": round(len(positive) / len(per_year), 3),
        "el_nino_years_positive": sum(fold["phase"] == "el_nino" for fold in positive),
        "la_nina_years_positive": sum(fold["phase"] == "la_nina" for fold in positive),
        "correlation_of_year_means": _correlation(
            [fold["mean_actual_pct"] for fold in per_year],
            [fold["mean_p50_pct"] for fold in per_year],
        ),
        "coverage_p10_p90": round(
            sum(fold["coverage_p10_p90"] * fold["rows"] for fold in per_year) / rows, 3
        ),
        "per_year": per_year,
    }


def verdict(metrics: Mapping[str, Any]) -> str:
    """Apply the rule fixed before training. Stated plainly so nobody oversells the model."""
    works = (
        metrics["pooled_skill"] >= MIN_POOLED_SKILL
        and metrics["share_years_positive"] >= MIN_SHARE_YEARS_POSITIVE
        and metrics["el_nino_years_positive"] >= 1
        and metrics["la_nina_years_positive"] >= 1
    )
    return "beats_climatology" if works else "no_skill_over_climatology"
