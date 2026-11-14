import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrofarm_ml.features import FEATURES, build_training_table
from agrofarm_ml.model import (
    QUANTILES,
    RainfallModel,
    climatology_quantiles,
    evaluate,
    pinball_loss,
    train_model,
)
from agrofarm_ml.scenario import simulate
from agrofarm_ml.splits import choose_holdout_years, season_phase_by_year, split_by_years
from agrofarm_ml.training import main as train_main


@pytest.fixture(scope="module")
def tables(climate: pd.DataFrame, enso: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = build_training_table(climate, enso)
    holdout = choose_holdout_years(season_phase_by_year(enso), years=sorted(set(table["year"])))
    return split_by_years(table, holdout)


@pytest.fixture(scope="module")
def model(tables: tuple[pd.DataFrame, pd.DataFrame]) -> RainfallModel:
    return train_model(tables[0], seed=0)


def test_pinball_loss_matches_a_hand_computed_case() -> None:
    actual = np.array([10.0, 10.0])
    predicted = np.array([8.0, 14.0])

    # under by 2 costs 0.9 * 2, over by 4 costs 0.1 * 4
    assert pinball_loss(actual, predicted, 0.9) == pytest.approx((1.8 + 0.4) / 2)


def test_predictions_have_no_nan_and_ordered_quantiles(
    model: RainfallModel, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    predicted = model.predict(tables[1][FEATURES])

    assert list(predicted.columns) == ["p10", "p50", "p90"]
    assert not predicted.isna().to_numpy().any()
    assert (predicted["p10"] <= predicted["p50"]).all()
    assert (predicted["p50"] <= predicted["p90"]).all()
    assert (predicted["p10"] >= 0).all()


def test_training_is_deterministic(tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    first = train_model(tables[0], seed=0).predict(tables[1][FEATURES])
    second = train_model(tables[0], seed=0).predict(tables[1][FEATURES])

    pd.testing.assert_frame_equal(first, second)


def test_model_beats_climatology_on_held_out_years_when_enso_matters(
    model: RainfallModel, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """CI gate on the fixture: the synthetic world has a strong ENSO effect, so a model that
    learned it must beat the calendar-month climatology on years it never saw."""
    metrics = evaluate(model, *tables)

    assert metrics["model"]["pinball_mean"] < metrics["climatology"]["pinball_mean"]
    assert metrics["skill_vs_climatology"] > 0.05
    assert 0.5 <= metrics["model"]["coverage_p10_p90"] <= 1.0
    assert set(metrics["holdout_years"]) == set(tables[1]["year"])


def test_climatology_is_computed_from_training_years_only(
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = tables
    clim = climatology_quantiles(train)
    july = train.loc[train["month_of_year"] == 7, "rainfall_mm"]

    assert clim.loc[7, "p50"] == pytest.approx(float(july.quantile(0.5)))
    assert sorted(clim.index) == list(range(1, 13))


def test_model_survives_a_save_and_load_round_trip(
    model: RainfallModel, tables: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    model.save(tmp_path)

    loaded = RainfallModel.load(tmp_path)

    pd.testing.assert_frame_equal(
        loaded.predict(tables[1][FEATURES]), model.predict(tables[1][FEATURES])
    )
    assert loaded.feature_ranges == model.feature_ranges


def test_el_nino_scenario_is_drier_than_la_nina_in_the_synthetic_world(
    model: RainfallModel,
) -> None:
    dry = simulate(model, month_of_year=7, enso_phase="el_nino")
    wet = simulate(model, month_of_year=7, enso_phase="la_nina")

    assert dry["scenario"]["p50"] < wet["scenario"]["p50"]
    assert dry["unit"] == "mm per month"
    assert dry["baseline"] == simulate(model, month_of_year=7, enso_phase="neutral")["scenario"]


def test_rainfall_change_scales_the_scenario_and_is_reported_as_an_assumption(
    model: RainfallModel,
) -> None:
    plain = simulate(model, month_of_year=7, enso_phase="neutral")
    drier = simulate(model, month_of_year=7, enso_phase="neutral", rainfall_change_pct=-20)

    assert drier["scenario"]["p50"] == pytest.approx(plain["scenario"]["p50"] * 0.8)
    assert any("rainfall_change_pct" in note for note in drier["assumptions"])


def test_inputs_outside_the_training_range_are_flagged_and_do_not_crash(
    model: RainfallModel,
) -> None:
    result = simulate(model, month_of_year=7, roni=9.0)

    assert "roni" in result["out_of_range_inputs"]
    assert not np.isnan(list(result["scenario"].values())).any()
    assert simulate(model, month_of_year=7, roni=0.2)["out_of_range_inputs"] == []


def test_invalid_scenario_inputs_are_rejected(model: RainfallModel) -> None:
    with pytest.raises(ValueError, match="month_of_year"):
        simulate(model, month_of_year=13, enso_phase="neutral")
    with pytest.raises(ValueError, match="enso_phase"):
        simulate(model, month_of_year=7, enso_phase="super_nino")
    with pytest.raises(ValueError, match="exactly one"):
        simulate(model, month_of_year=7)


def test_training_cli_writes_the_model_and_a_model_card(
    climate: pd.DataFrame, enso: pd.DataFrame, tmp_path: Path
) -> None:
    climate.to_parquet(tmp_path / "climate.parquet", index=False)
    enso.to_parquet(tmp_path / "enso.parquet", index=False)
    out = tmp_path / "models" / "rainfall" / "test"

    code = train_main(
        ["--climate", str(tmp_path / "climate.parquet"), "--enso", str(tmp_path / "enso.parquet"),
         "--out", str(out)]
    )  # fmt: skip

    card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
    assert code == 0
    assert card["target"] == "monthly rainfall total at the station (mm per month)"
    assert card["quantiles"] == list(QUANTILES)
    assert card["features"] == FEATURES
    assert card["validation"]["split"] == "whole years"
    assert {"el_nino", "la_nina"} <= set(card["validation"]["holdout_phases"].values())
    assert card["metrics"]["model"]["pinball_mean"] > 0
    assert len(card["data_versions"]["climate_sha256"]) == 64
    assert card["limits"]
    assert RainfallModel.load(out).predict(
        build_training_table(climate, enso)[FEATURES].head(3)
    ).shape == (3, 3)
