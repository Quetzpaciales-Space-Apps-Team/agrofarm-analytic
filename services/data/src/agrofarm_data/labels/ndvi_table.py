"""Cell and year table for the experimental NDVI model.

One row per grid cell and year: the seasonal peak NDVI on cropland (the proxy label), rainfall
totals of the rainy season, its mean temperature and the ENSO state. Rows with any missing input
are left out; a missing value is never replaced by zero or by a guess.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SEASON = (5, 10)  # May to October, the rainy season
EARLY = (5, 7)
LATE = (8, 10)
PRE_SEASON = (2, 4)  # February to April: the ENSO state before sowing

COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
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


@dataclass(frozen=True)
class CellGrid:
    """Regular latitude and longitude grid. Row 0 is the northern edge."""

    west: float
    north: float
    resolution: float
    rows: int
    cols: int

    def latitudes(self) -> NDArray[np.float64]:
        return np.asarray(self.north - (np.arange(self.rows) + 0.5) * self.resolution, "float64")

    def longitudes(self) -> NDArray[np.float64]:
        return np.asarray(self.west + (np.arange(self.cols) + 0.5) * self.resolution, "float64")


@dataclass
class TableInputs:
    peaks: dict[int, NDArray[Any]]  # year -> seasonal peak NDVI per cell
    cropland: NDArray[Any]  # mean cropland fraction per cell
    rain: NDArray[Any]  # (month, row, col) monthly rainfall in mm
    rain_months: NDArray[Any]  # YYYYMM of each rain layer
    temperature: dict[int, NDArray[Any]]  # year -> season mean temperature per cell, deg C
    enso: pd.DataFrame  # columns month (datetime) and roni


def _season_sum(inputs: TableInputs, year: int, months: tuple[int, int]) -> NDArray[np.float64]:
    wanted = [year * 100 + month for month in range(months[0], months[1] + 1)]
    picked = np.isin(inputs.rain_months, wanted)
    if int(picked.sum()) != len(wanted):
        return np.full(inputs.rain.shape[1:], np.nan)
    # A plain sum: one missing month makes the total missing.
    total: NDArray[np.float64] = inputs.rain[picked].astype("float64").sum(axis=0)
    return total


def _roni(enso: pd.DataFrame, year: int, months: tuple[int, int]) -> float:
    month = enso["month"].dt
    rows = enso[(month.year == year) & month.month.between(*months)]
    if len(rows) != months[1] - months[0] + 1 or rows["roni"].isna().any():
        return float("nan")
    return float(rows["roni"].mean())


def build_cell_year_table(
    grid: CellGrid, inputs: TableInputs, min_cropland_fraction: float = 0.05
) -> pd.DataFrame:
    latitudes, longitudes = grid.latitudes(), grid.longitudes()
    lon, lat = np.meshgrid(longitudes, latitudes)
    ids = np.array([[f"{la:.3f}_{lo:.3f}" for lo in longitudes] for la in latitudes], dtype=object)
    cropland = np.nan_to_num(np.asarray(inputs.cropland, dtype="float64"), nan=0.0)
    frames = []
    for year in sorted(inputs.peaks):
        if year not in inputs.temperature:
            continue
        frame = pd.DataFrame(
            {
                "cell_id": ids.ravel(),
                "latitude": lat.ravel(),
                "longitude": lon.ravel(),
                "year": year,
                "ndvi_peak": np.asarray(inputs.peaks[year], dtype="float64").ravel(),
                "rain_season_mm": _season_sum(inputs, year, SEASON).ravel(),
                "rain_early_mm": _season_sum(inputs, year, EARLY).ravel(),
                "rain_late_mm": _season_sum(inputs, year, LATE).ravel(),
                "temp_season_c": np.asarray(inputs.temperature[year], dtype="float64").ravel(),
                "roni_season": _roni(inputs.enso, year, SEASON),
                "roni_pre_season": _roni(inputs.enso, year, PRE_SEASON),
                "cropland_fraction": cropland.ravel(),
            }
        )
        frames.append(frame[frame["cropland_fraction"] >= min_cropland_fraction])
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return table[COLUMNS].astype({"cell_id": "string", "year": "int64"})
