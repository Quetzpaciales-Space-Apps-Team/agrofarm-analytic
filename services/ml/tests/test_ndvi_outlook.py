"""Serving NDVI preserves its failed evaluation result and its vegetation target."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agrofarm_ml.main import create_app
from agrofarm_ml.ndvi.model import train_model
from agrofarm_ml.ndvi.outlook import NdviOutlook
from agrofarm_ml.ndvi.table import with_anomalies
from tests.test_ndvi_model import _table


def make_outlook() -> NdviOutlook:
    table = with_anomalies(_table(0.3), _table(0.3))
    row = table[table["cell_id"] == "c0"].iloc[0]
    cell = {
        "cell_id": "c0",
        "latitude": 14.575,
        "longitude": -90.525,
        **{
            k: float(row[k])
            for k in ("ndvi_normal", "rain_normal_mm", "temp_normal_c", "cropland_fraction")
        },
    }
    evaluation: dict[str, Any] = {
        "model_version": "archived-test-model",
        "verdict": "no_skill_over_climatology",
        "metrics": {
            "pooled_skill": 0.0441,
            "correlation_of_year_means": 0.127,
            "coverage_p10_p90": 0.756,
            "years": 25,
        },
        "data_versions": {"table_sha256": "test-only"},
    }
    return NdviOutlook(train_model(table), [cell], evaluation)


@pytest.fixture(scope="module")
def outlook() -> NdviOutlook:
    return make_outlook()


def test_same_scenario_matches_baseline_and_changed_rain_changes_ndvi(outlook: NdviOutlook) -> None:
    neutral = outlook.simulate(14.59, -90.53, 0, 0, "neutral")
    changed = outlook.simulate(14.59, -90.53, 40, 1, "el_nino")
    assert neutral["baseline"] == neutral["scenario"]
    assert changed["scenario"]["p50"] != changed["baseline"]["p50"]
    assert changed["target"] == "seasonal_peak_ndvi_anomaly_pct"
    assert changed["unit"] == "%"
    assert changed["validation"]["acceptance_rule_met"] is False
    assert changed["validation"]["skill_improvement_pct"] == 4.41
    assert changed["provenance"]["evaluation_verdict"] == "no_skill_over_climatology"


def test_quantiles_are_ordered_extremes_flagged_and_shap_delta_is_additive(
    outlook: NdviOutlook,
) -> None:
    result = outlook.simulate(14.59, -90.53, 100, 8, "la_nina")
    for key in ["baseline", "scenario"]:
        q = result[key]
        assert q["p10"] <= q["p50"] <= q["p90"]
    assert "rain_anom_pct" in result["out_of_range_inputs"]
    assert sum(item["contribution"] for item in result["attributions"]) == pytest.approx(
        result["explained_median_delta"], abs=1e-4
    )
    assert result["reference_cell"]["distance_km"] < 15


def test_no_estimate_is_invented_for_locations_without_a_trained_cell(outlook: NdviOutlook) -> None:
    with pytest.raises(ValueError, match="cropland cell"):
        outlook.simulate(17.5, -89.0, 0, 0, "neutral")


def test_http_route_has_its_own_target_and_missing_model_is_503(
    outlook: NdviOutlook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NDVI_MODEL_DIR", raising=False)
    payload = {
        "farm_id": "22222222-2222-4222-8222-222222222222",
        "centroid_lat": 14.59,
        "centroid_lon": -90.53,
        "scenario": {"rainfall_change_pct": 0, "temperature_change_c": 0, "enso_phase": "neutral"},
    }
    assert TestClient(create_app()).post("/v1/vegetation/simulate", json=payload).status_code == 503
    client = TestClient(create_app(vegetation_model=outlook))
    response = client.post("/v1/vegetation/simulate", json=payload)
    assert response.status_code == 200
    assert response.json()["target"] == "seasonal_peak_ndvi_anomaly_pct"
    assert (
        client.post("/v1/vegetation/simulate", json={**payload, "centroid_lat": 40}).status_code
        == 422
    )


def test_pinned_artifact_loads_and_rejects_changed_weights(
    outlook: NdviOutlook, tmp_path: Any
) -> None:
    import hashlib
    import json

    outlook.model.save(tmp_path)
    (tmp_path / "reference_cells.json").write_text(json.dumps(outlook.cells))
    (tmp_path / "model_card.json").write_text(json.dumps(outlook.evaluation))
    card: dict[str, Any] = {
        "model_version": "ndvi-vegetation-outlook-0.1.0",
        "owner_accepted_limited_use": True,
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()
        },
    }
    path = tmp_path / "serving_card.json"
    path.write_text(json.dumps(card))
    loaded = NdviOutlook.load(tmp_path)
    assert (
        loaded.simulate(14.59, -90.53, 10, 1, "neutral")["validation"]["acceptance_rule_met"]
        is False
    )
    (tmp_path / "p50.lgb.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        NdviOutlook.load(tmp_path)
    card["files_sha256"].pop("p50.lgb.txt")
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="manifest"):
        NdviOutlook.load(tmp_path)
    card["owner_accepted_limited_use"] = False
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="acceptance"):
        NdviOutlook.load(tmp_path)
