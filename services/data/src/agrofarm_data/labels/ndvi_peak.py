"""Seasonal peak NDVI on cropland per grid cell: the proxy label of the experimental model.

NDVI is not yield. It says how green the cropland of a cell got during the rainy season. The
functions here are pure array maths; reading files is left to the caller.

Steps: a MODIS composite on its sinusoidal grid is sampled at the centres of a fine latitude and
longitude grid, the composites of one season are reduced to their maximum (clouds lower NDVI, so
the maximum keeps the clearest date), and the fine grid is averaged into coarse cells with the
cropland fraction as weight.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_M = 6371007.181  # radius of the MODIS sinusoidal sphere
NDVI_SCALE = 0.0001
VALID_RAW_RANGE = (-2000, 10000)  # MOD13 valid range; the fill value (-3000) is outside it

Float = NDArray[np.float64]


class LabelError(ValueError):
    """The inputs cannot give a label."""


def sinusoidal_xy(latitude: NDArray[Any], longitude: NDArray[Any]) -> tuple[Float, Float]:
    """Latitude and longitude in degrees to MODIS sinusoidal metres."""
    lat = np.radians(np.asarray(latitude, dtype="float64"))
    lon = np.radians(np.asarray(longitude, dtype="float64"))
    return EARTH_RADIUS_M * lon * np.cos(lat), EARTH_RADIUS_M * lat


def sample_sinusoidal(
    raw: NDArray[Any], transform: Sequence[float], x: NDArray[Any], y: NDArray[Any]
) -> Float:
    """NDVI of the pixel under each point. NaN for fill, out-of-range values and no pixel.

    `transform` is the affine of `raw` as (a, b, c, d, e, f): x = a*col + c, y = e*row + f.
    """
    a, _, c, _, e, f = (float(value) for value in transform[:6])
    col = np.floor((np.asarray(x, dtype="float64") - c) / a).astype(np.int64)
    row = np.floor((np.asarray(y, dtype="float64") - f) / e).astype(np.int64)
    inside = (row >= 0) & (row < raw.shape[0]) & (col >= 0) & (col < raw.shape[1])
    values = np.full(col.shape, np.nan, dtype="float64")
    picked = raw[row[inside], col[inside]].astype("float64")
    low, high = VALID_RAW_RANGE
    picked[(picked < low) | (picked > high)] = np.nan
    values[inside] = picked * NDVI_SCALE
    return values


def seasonal_peak(stack: NDArray[Any], min_composites: int) -> Float:
    """Maximum over the composites of a season (axis 0). NaN where too few were usable."""
    data = np.asarray(stack, dtype="float64")
    usable = (~np.isnan(data)).sum(axis=0)
    peak = np.where(np.isnan(data), -np.inf, data).max(axis=0)
    return np.where(usable >= min_composites, peak, np.nan)


def _blocks(values: Float, factor: int) -> Float:
    rows, cols = values.shape
    return values.reshape(rows // factor, factor, cols // factor, factor)


def cropland_weighted_cells(
    peak: NDArray[Any], cropland_fraction: NDArray[Any], factor: int, min_cover: float
) -> tuple[Float, Float]:
    """Average `factor` by `factor` fine cells into one, weighted by cropland fraction.

    Returns the cell value and the mean cropland fraction of the cell. A cell has no value when
    it has no cropland, or when less than `min_cover` of its cropland weight was observed.
    Cropland is a weight, never a yes-or-no mask.
    """
    values = np.asarray(peak, dtype="float64")
    fraction = np.asarray(cropland_fraction, dtype="float64")
    if values.shape != fraction.shape:
        raise LabelError(f"Shapes differ: {values.shape} and {fraction.shape}.")
    if values.shape[0] % factor or values.shape[1] % factor:
        raise LabelError(f"Grid {values.shape} does not divide into cells of {factor}.")
    weight = np.nan_to_num(fraction, nan=0.0)
    observed = np.where(np.isnan(values), 0.0, weight)
    total = _blocks(weight, factor).sum(axis=(1, 3))
    seen = _blocks(observed, factor).sum(axis=(1, 3))
    weighted = _blocks(np.nan_to_num(values, nan=0.0) * observed, factor).sum(axis=(1, 3))
    enough = (total > 0) & (seen >= min_cover * total)
    with np.errstate(invalid="ignore", divide="ignore"):
        cells = np.where(enough, weighted / seen, np.nan)
    return cells, total / float(factor * factor)
