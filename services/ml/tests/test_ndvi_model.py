"""Experimental model: seasonal peak NDVI anomaly on cropland, one whole year held out."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrofarm_ml.ndvi.model import (
    FEATURES,
    TARGET,
    NdviModel,
    leave_one_year_out,
    train_model,
    verdict,
)
from agrofarm_ml.ndvi.table import TableError, with_anomalies
from agrofarm_ml.ndvi.training import main

YEARS = list(range(2000, 2016))


def _table(effect: float, seed: int = 0, cells: int = 40) -> pd.DataFrame:
    """Synthetic cell and year table. `effect` is how much rain moves the peak NDVI."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in YEARS:
        year_rain = rng.normal(0.0, 0.25)
        roni = float(rng.normal(0.0, 0.8))
        for cell in range(cells):
            normal_rain = 900.0 + 20.0 * cell
            rain = normal_rain * (1.0 + year_rain + rng.normal(0.0, 0.05))
            ndvi = 0.6 + 0.002 * cell
            ndvi *= 1.0 + effect * (rain / normal_rain - 1.0) + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "cell_id": f"c{cell}",
                    "year": year,
                    "ndvi_peak": ndvi,
                    "rain_season_mm": rain,
                    "rain_early_mm": rain * 0.45,
                    "rain_late_mm": rain * 0.55,
                    "temp_season_c": 22.0 + rng.normal(0.0, 0.3),
                    "roni_season": roni,
                    "roni_pre_season": roni * 0.5,
                    "cropland_fraction": 0.1 + 0.01 * cell,
                }
            )
    return pd.DataFrame(rows)


def _enso() -> pd.DataFrame:
    """Monthly RONI with El Nino, La Nina and neutral rainy seasons among the years."""
    months = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-01", freq="MS")
    by_year = {year: (1.0, -1.0, 0.0)[index % 3] for index, year in enumerate(YEARS)}
    return pd.DataFrame({"month": months, "roni": [by_year[m.year] for m in months]})


def test_anomalies_use_only_the_reference_rows() -> None:
    table = _table(effect=0.3)
    reference = table[table["year"] != 2015]

    result = with_anomalies(table, reference)
    changed = table.copy()
    changed.loc[changed["year"] == 2015, "ndvi_peak"] *= 3.0
    again = with_anomalies(changed, reference)

    # Changing the held-out year must not move the normal of any cell.
    assert result["ndvi_normal"].equals(again["ndvi_normal"])
    one = result[(result["cell_id"] == "c0") & (result["year"] == 2000)].iloc[0]
    normal = reference[reference["cell_id"] == "c0"]["ndvi_peak"].mean()
    assert one["ndvi_normal"] == pytest.approx(normal)
    assert one[TARGET] == pytest.approx(100.0 * (one["ndvi_peak"] / normal - 1.0))


def test_cells_with_too_few_reference_years_are_dropped() -> None:
    table = _table(effect=0.3)
    reference = table[(table["cell_id"] != "c1") | (table["year"] < 2003)]

    result = with_anomalies(table, reference, min_reference_years=10)

    assert "c1" not in set(result["cell_id"])
    assert "c0" in set(result["cell_id"])


def test_missing_columns_are_named() -> None:
    table = _table(effect=0.3).drop(columns=["rain_late_mm"])
    with pytest.raises(TableError, match="rain_late_mm"):
        with_anomalies(table, table)


def test_missing_values_are_rejected_not_filled() -> None:
    table = _table(effect=0.3)
    table.loc[3, "ndvi_peak"] = np.nan
    with pytest.raises(TableError, match="missing"):
        with_anomalies(table, table)


def test_predictions_are_ordered_quantiles_and_the_model_round_trips(tmp_path: Path) -> None:
    table = _table(effect=0.3)
    train = with_anomalies(table, table)
    model = train_model(train)

    predicted = model.predict(train[FEATURES])
    model.save(tmp_path)
    loaded = NdviModel.load(tmp_path)

    assert (predicted["p10"] <= predicted["p50"]).all()
    assert (predicted["p50"] <= predicted["p90"]).all()
    assert not predicted.isna().any().any()
    pd.testing.assert_frame_equal(predicted, loaded.predict(train[FEATURES]))
    assert set(loaded.feature_ranges) == set(FEATURES)


def test_training_is_deterministic() -> None:
    train = with_anomalies(_table(effect=0.3), _table(effect=0.3))
    first = train_model(train, seed=3).predict(train[FEATURES])
    second = train_model(train, seed=3).predict(train[FEATURES])
    pd.testing.assert_frame_equal(first, second)


def test_every_year_is_predicted_by_a_model_that_never_saw_it() -> None:
    table = _table(effect=0.3)
    phases = pd.Series({year: "neutral" for year in YEARS})
    seen: list[tuple[set[int], set[int]]] = []

    leave_one_year_out(
        table, phases, on_fold=lambda train, test: seen.append((set(train), set(test)))
    )

    assert len(seen) == len(YEARS)
    for train_years, test_years in seen:
        assert len(test_years) == 1
        assert not train_years & test_years


def test_a_real_signal_beats_climatology_and_noise_does_not() -> None:
    phases = pd.Series(
        {year: ("el_nino", "la_nina", "neutral")[i % 3] for i, year in enumerate(YEARS)}
    )

    signal = leave_one_year_out(_table(effect=0.3), phases)
    noise = leave_one_year_out(_table(effect=0.0), phases)

    assert signal["pooled_skill"] > 0.2
    assert verdict(signal) == "beats_climatology"
    assert noise["pooled_skill"] < 0.05
    assert verdict(noise) == "no_skill_over_climatology"


def test_the_verdict_needs_every_condition() -> None:
    good = {
        "pooled_skill": 0.2,
        "share_years_positive": 0.8,
        "el_nino_years_positive": 2,
        "la_nina_years_positive": 1,
    }
    assert verdict(good) == "beats_climatology"
    assert verdict({**good, "pooled_skill": 0.04}) == "no_skill_over_climatology"
    assert verdict({**good, "share_years_positive": 0.5}) == "no_skill_over_climatology"
    assert verdict({**good, "el_nino_years_positive": 0}) == "no_skill_over_climatology"
    assert verdict({**good, "la_nina_years_positive": 0}) == "no_skill_over_climatology"


def test_training_cli_writes_a_model_and_a_card_that_says_what_it_is(tmp_path: Path) -> None:
    table_path, enso_path, out = (
        tmp_path / "table.parquet",
        tmp_path / "enso.parquet",
        tmp_path / "m",
    )
    _table(effect=0.3).to_parquet(table_path, index=False)
    _enso().to_parquet(enso_path, index=False)

    code = main(["--table", str(table_path), "--enso", str(enso_path), "--out", str(out)])

    card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
    assert code == 0
    assert card["status"] == "experimental"
    assert card["target"].startswith("seasonal peak NDVI")
    assert card["verdict"] in {"beats_climatology", "no_skill_over_climatology"}
    assert card["validation"]["split"] == "leave one whole year out"
    assert card["validation"]["rule"]
    assert any("not yield" in limit.lower() for limit in card["limits"])
    assert len(card["metrics"]["per_year"]) == len(YEARS)
    assert (out / "p50.lgb.txt").exists()
