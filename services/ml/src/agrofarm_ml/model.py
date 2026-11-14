"""Baseline rainfall model: gradient boosting with quantile regression (p10, p50, p90).

Target: monthly rainfall total at the station, in mm. The benchmark every model must beat
is the calendar-month climatology of the training years.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agrofarm_ml.features import FEATURES, TARGET

QUANTILES = (0.1, 0.5, 0.9)
NAMES = ["p10", "p50", "p90"]
MODEL_VERSION = "rainfall-baseline-0.1.0"
PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 4,
    "min_data_in_leaf": 5,
    "min_data_in_bin": 1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
ROUNDS = 150


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


@dataclass
class RainfallModel:
    boosters: dict[str, lgb.Booster]
    feature_ranges: dict[str, tuple[float, float]]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = features[FEATURES].to_numpy(dtype=float)
        raw = np.column_stack([self.boosters[name].predict(matrix) for name in NAMES])
        # Independent quantile models can cross. Sorting restores p10 <= p50 <= p90.
        ordered = np.sort(np.clip(raw, 0.0, None), axis=1)
        return pd.DataFrame(ordered, columns=NAMES, index=features.index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, booster in self.boosters.items():
            booster.save_model(str(directory / f"{name}.lgb.txt"))
        (directory / "feature_ranges.json").write_text(
            json.dumps(self.feature_ranges, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "RainfallModel":
        boosters = {
            name: lgb.Booster(model_file=str(directory / f"{name}.lgb.txt")) for name in NAMES
        }
        ranges = json.loads((directory / "feature_ranges.json").read_text(encoding="utf-8"))
        return cls(boosters, {key: (low, high) for key, (low, high) in ranges.items()})


def train_model(train: pd.DataFrame, seed: int = 0) -> RainfallModel:
    matrix = train[FEATURES].to_numpy(dtype=float)
    target = train[TARGET].to_numpy(dtype=float)
    boosters = {}
    for name, quantile in zip(NAMES, QUANTILES, strict=True):
        params = {**PARAMS, "alpha": quantile, "seed": seed}
        boosters[name] = lgb.train(params, lgb.Dataset(matrix, label=target), ROUNDS)
    ranges = {
        column: (float(train[column].min()), float(train[column].max())) for column in FEATURES
    }
    return RainfallModel(boosters, ranges)


def climatology_quantiles(train: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: quantiles of each calendar month, from the training years only."""
    grouped = train.groupby("month_of_year")[TARGET]
    return pd.DataFrame(
        {name: grouped.quantile(q) for name, q in zip(NAMES, QUANTILES, strict=True)}
    )


def _score(actual: np.ndarray, predicted: pd.DataFrame) -> dict[str, float]:
    losses = {
        name: pinball_loss(actual, predicted[name].to_numpy(), q)
        for name, q in zip(NAMES, QUANTILES, strict=True)
    }
    inside = (actual >= predicted["p10"].to_numpy()) & (actual <= predicted["p90"].to_numpy())
    return {
        **{f"pinball_{name}": round(value, 3) for name, value in losses.items()},
        "pinball_mean": round(float(np.mean(list(losses.values()))), 3),
        "mae_p50": round(float(np.mean(np.abs(actual - predicted["p50"].to_numpy()))), 3),
        "coverage_p10_p90": round(float(inside.mean()), 3),
    }


def evaluate(model: RainfallModel, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Model against climatology on the held-out years. Units: mm per month."""
    actual = test[TARGET].to_numpy(dtype=float)
    clim = climatology_quantiles(train).reindex(test["month_of_year"]).reset_index(drop=True)
    model_score = _score(actual, model.predict(test[FEATURES]).reset_index(drop=True))
    clim_score = _score(actual, clim)
    skill = 1.0 - model_score["pinball_mean"] / clim_score["pinball_mean"]
    return {
        "holdout_years": sorted(int(year) for year in set(test["year"])),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model": model_score,
        "climatology": clim_score,
        "skill_vs_climatology": round(float(skill), 3),
    }
