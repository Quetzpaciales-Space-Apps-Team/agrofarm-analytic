"""Cell and year table for the experimental NDVI model: label, rainfall, temperature and ENSO."""

import numpy as np
import pandas as pd
import pytest

from agrofarm_data.labels.ndvi_table import CellGrid, TableInputs, build_cell_year_table

NAN = float("nan")
GRID = CellGrid(west=-91.0, north=15.0, resolution=0.5, rows=1, cols=2)


def _inputs(**changes: object) -> TableInputs:
    months = np.array([y * 100 + m for y in (2001, 2002) for m in range(1, 13)])
    rain = np.ones((24, 1, 2)) * 100.0
    rain[12:] = 50.0  # 2002 is drier
    enso = pd.DataFrame(
        {
            "month": pd.date_range("2001-01-01", "2002-12-01", freq="MS"),
            "roni": [1.0] * 12 + [-1.0] * 12,
        }
    )
    base = {
        "peaks": {2001: np.array([[0.8, 0.6]]), 2002: np.array([[0.7, NAN]])},
        "cropland": np.array([[0.4, 0.3]]),
        "rain": rain,
        "rain_months": months,
        "temperature": {2001: np.array([[22.0, 23.0]]), 2002: np.array([[22.5, 23.5]])},
        "enso": enso,
    }
    return TableInputs(**{**base, **changes})  # type: ignore[arg-type]


def test_one_row_per_cell_and_year_with_season_totals_and_enso() -> None:
    table = build_cell_year_table(GRID, _inputs())

    assert list(table["year"]) == [2001, 2001, 2002]  # the cell without a label has no row
    first = table.iloc[0]
    assert first["cell_id"] == "14.750_-90.750"
    assert first["latitude"] == pytest.approx(14.75)
    assert first["longitude"] == pytest.approx(-90.75)
    assert first["ndvi_peak"] == pytest.approx(0.8)
    assert first["rain_season_mm"] == pytest.approx(600.0)  # May to October
    assert first["rain_early_mm"] == pytest.approx(300.0)  # May to July
    assert first["rain_late_mm"] == pytest.approx(300.0)  # August to October
    assert first["temp_season_c"] == pytest.approx(22.0)
    assert first["roni_season"] == pytest.approx(1.0)
    assert first["roni_pre_season"] == pytest.approx(1.0)
    assert first["cropland_fraction"] == pytest.approx(0.4)
    assert table.iloc[2]["rain_season_mm"] == pytest.approx(300.0)
    assert table.iloc[2]["roni_season"] == pytest.approx(-1.0)


def test_cells_under_the_cropland_minimum_are_left_out() -> None:
    table = build_cell_year_table(GRID, _inputs(), min_cropland_fraction=0.35)
    assert set(table["cell_id"]) == {"14.750_-90.750"}


def test_a_missing_rain_month_drops_the_row_and_is_never_counted_as_zero() -> None:
    inputs = _inputs()
    inputs.rain[6, 0, 0] = NAN  # July 2001, first cell

    table = build_cell_year_table(GRID, inputs)

    assert not ((table["cell_id"] == "14.750_-90.750") & (table["year"] == 2001)).any()
    assert ((table["cell_id"] == "14.750_-90.750") & (table["year"] == 2002)).any()
    assert not table.isna().any().any()


def test_a_year_without_temperature_or_enso_has_no_rows() -> None:
    inputs = _inputs(temperature={2001: np.array([[22.0, 23.0]])})
    table = build_cell_year_table(GRID, inputs)
    assert set(table["year"]) == {2001}
