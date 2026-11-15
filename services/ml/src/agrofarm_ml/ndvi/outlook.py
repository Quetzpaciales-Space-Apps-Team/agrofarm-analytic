"""Conditional regional vegetation outlook; historical acceptance failure is preserved."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agrofarm_ml.ndvi.model import FEATURES, NdviModel

VERSION = "ndvi-vegetation-outlook-0.1.0"
LIMITS = [
    "Regional mixed cropland at approximately 0.05 degrees; not parcel yield or crop growth.",
    "Conditional scenarios, not an ENSO or seasonal weather forecast.",
    "The original 5% improvement goal was not reached; annual mean correlation was 0.127.",
    "Rainfall changes apply uniformly to early, late and full seasons. "
    "ENSO phases use RONI +1, 0, -1 in both periods.",
    "SHAP differences explain the raw median regressor, not causation. "
    "Sorted quantiles can change its position.",
]


class NdviOutlook:
    def __init__(self, model: NdviModel, cells: list[dict[str, Any]], evaluation: dict[str, Any]):
        self.model = model
        self.cells = cells
        self.evaluation = evaluation

    @classmethod
    def load(cls, directory: Path) -> "NdviOutlook":
        card = json.loads((directory / "serving_card.json").read_text())
        if card["model_version"] != VERSION or card["owner_accepted_limited_use"] is not True:
            raise ValueError("NDVI limited-use acceptance missing")
        expected = {
            "p10.lgb.txt",
            "p50.lgb.txt",
            "p90.lgb.txt",
            "feature_ranges.json",
            "reference_cells.json",
            "model_card.json",
        }
        if set(card["files_sha256"]) != expected:
            raise ValueError("NDVI artifact manifest mismatch")
        for name, checksum in card["files_sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("NDVI artifact checksum mismatch")
        model = NdviModel.load(directory)
        if set(model.feature_ranges) != set(FEATURES):
            raise ValueError("NDVI feature mismatch")
        return cls(
            model,
            json.loads((directory / "reference_cells.json").read_text()),
            json.loads((directory / "model_card.json").read_text()),
        )

    def simulate(
        self, latitude: float, longitude: float, rain: float, temperature: float, phase: str
    ) -> dict[str, Any]:
        def distance(cell: dict[str, Any]) -> float:
            lat1, lat2 = math.radians(latitude), math.radians(cell["latitude"])
            delta = math.radians(longitude - cell["longitude"])
            a = (
                math.sin((lat1 - lat2) / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(delta / 2) ** 2
            )
            return 6371 * 2 * math.asin(min(1, math.sqrt(a)))

        cell = min(self.cells, key=distance)
        km = distance(cell)
        if km > 15:
            raise ValueError("No trained cropland cell within 15 km")
        normal = {
            name: float(cell[name])
            for name in ("cropland_fraction", "ndvi_normal", "rain_normal_mm", "temp_normal_c")
        }
        baseline = {**normal, **dict.fromkeys(FEATURES[:6], 0.0)}
        roni = {"el_nino": 1.0, "neutral": 0.0, "la_nina": -1.0}[phase]
        scenario = {
            **normal,
            "rain_anom_pct": rain,
            "rain_early_anom_pct": rain,
            "rain_late_anom_pct": rain,
            "temp_anom_c": temperature,
            "roni_season": roni,
            "roni_pre_season": roni,
        }
        frame = pd.DataFrame([baseline, scenario])[FEATURES]
        quantiles = self.model.predict(frame)
        shap = np.asarray(self.model.boosters["p50"].predict(frame.to_numpy(), pred_contrib=True))
        delta = shap[1, :-1] - shap[0, :-1]
        metrics = self.evaluation["metrics"]
        return {
            "target": "seasonal_peak_ndvi_anomaly_pct",
            "unit": "%",
            "baseline": quantiles.iloc[0].to_dict(),
            "scenario": quantiles.iloc[1].to_dict(),
            "reference_cell": {
                "cell_id": cell["cell_id"],
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "distance_km": round(km, 3),
                "resolution_degrees": 0.05,
            },
            "attributions": [
                {"feature": name, "contribution": float(value)}
                for name, value in zip(FEATURES, delta, strict=True)
            ],
            "explained_median_delta": float(delta.sum()),
            "out_of_range_inputs": [
                name
                for name in FEATURES
                if not self.model.feature_ranges[name][0]
                <= scenario[name]
                <= self.model.feature_ranges[name][1]
            ],
            "validation": {
                "acceptance_rule_met": False,
                "skill_improvement_pct": round(metrics["pooled_skill"] * 100, 2),
                "required_improvement_pct": 5,
                "annual_mean_correlation": metrics["correlation_of_year_means"],
                "interval_coverage": metrics["coverage_p10_p90"],
                "years": metrics["years"],
            },
            "limitations": LIMITS,
            "needs_local_review": True,
            "provenance": {
                "model_version": VERSION,
                "evaluation_model_version": self.evaluation["model_version"],
                "evaluation_verdict": self.evaluation["verdict"],
                "data_versions": self.evaluation.get("data_versions", {}),
            },
        }
