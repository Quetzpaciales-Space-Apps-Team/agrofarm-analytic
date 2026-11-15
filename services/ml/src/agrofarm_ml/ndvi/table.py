"""Anomalies of the cell and year table against each cell's own normal.

Input: one row per grid cell and year with the seasonal peak NDVI on cropland, rainfall totals,
mean temperature and the ENSO state (written by data-service). The normal of a cell is its mean
over the reference rows only, so a held-out year never leaks into its own anomaly.
"""

import pandas as pd

TARGET = "ndvi_peak_anom_pct"
COLUMNS = [
    "cell_id",
    "year",
    "ndvi_peak",
    "rain_season_mm",
    "rain_early_mm",
    "rain_late_mm",
    "temp_season_c",
    "roni_season",
    "roni_pre_season",
    "cropland_fraction",
]


class TableError(ValueError):
    """The table cannot be used for training."""


def _check(table: pd.DataFrame) -> None:
    absent = [column for column in COLUMNS if column not in table.columns]
    if absent:
        raise TableError(f"Columns not found: {absent}.")
    if table[COLUMNS].isna().any().any():
        raise TableError("The table has missing values. Drop those rows; they are never filled.")


def _pct(value: "pd.Series[float]", normal: "pd.Series[float]") -> "pd.Series[float]":
    return 100.0 * (value / normal - 1.0)


def with_anomalies(
    table: pd.DataFrame, reference: pd.DataFrame, min_reference_years: int = 10
) -> pd.DataFrame:
    """Add the normals and anomalies. Cells with too few reference years are dropped."""
    _check(table)
    _check(reference)
    grouped = reference.groupby("cell_id")
    normals = pd.DataFrame(
        {
            "ndvi_normal": grouped["ndvi_peak"].mean(),
            "rain_normal_mm": grouped["rain_season_mm"].mean(),
            "rain_early_normal_mm": grouped["rain_early_mm"].mean(),
            "rain_late_normal_mm": grouped["rain_late_mm"].mean(),
            "temp_normal_c": grouped["temp_season_c"].mean(),
            "reference_years": grouped["year"].nunique(),
        }
    )
    result = table.join(normals, on="cell_id")
    result = result[result["reference_years"] >= min_reference_years].reset_index(drop=True)
    result[TARGET] = _pct(result["ndvi_peak"], result["ndvi_normal"])
    result["rain_anom_pct"] = _pct(result["rain_season_mm"], result["rain_normal_mm"])
    result["rain_early_anom_pct"] = _pct(result["rain_early_mm"], result["rain_early_normal_mm"])
    result["rain_late_anom_pct"] = _pct(result["rain_late_mm"], result["rain_late_normal_mm"])
    result["temp_anom_c"] = result["temp_season_c"] - result["temp_normal_c"]
    return result
