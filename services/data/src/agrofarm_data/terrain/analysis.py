"""Slope, aspect, depression filling and D8 flow accumulation on a DEM in EPSG:4326.

The grid is in degrees, so every cell size is converted to metres from the latitude of its row
before any gradient is taken. A metre of longitude shrinks with cos(latitude); a degree of
latitude is taken from the WGS84 series, a degree of longitude as cos(latitude) times that
(a spherical approximation, off by well under 1 percent for slope purposes).

Arrays are float64 with NaN for no data. Rows run north to south.
"""

import heapq
import math

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine

Grid = NDArray[np.float64]

# Below this slope the surface counts as flat and the aspect is undefined (null, not 0 or 360).
FLAT_SLOPE_DEG = 0.01

# The eight D8 neighbours as (row offset, column offset).
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def metres_per_degree(latitude: NDArray[np.float64]) -> tuple[Grid, Grid]:
    """Metres per degree of (longitude, latitude) at each latitude in degrees."""
    phi = np.radians(latitude)
    lat_m = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    return np.cos(phi) * lat_m, lat_m


def cell_sizes_m(transform: Affine, rows: int) -> tuple[Grid, Grid]:
    """Cell width and height in metres for each row (east-west size, north-south size)."""
    latitude = transform.f + (np.arange(rows) + 0.5) * transform.e
    lon_m, lat_m = metres_per_degree(latitude)
    return abs(transform.a) * lon_m, abs(transform.e) * lat_m


def slope_aspect(dem: Grid, transform: Affine) -> tuple[Grid, Grid]:
    """Slope in degrees and aspect in degrees clockwise from north (the facing direction).

    Horn's method on the 3 x 3 neighbourhood. A cell whose neighbourhood holds no data is NaN.
    The outermost ring of the array extends the surface linearly (one-sided difference), so read
    a margin around the area of interest for a better estimate there. Aspect is NaN where the
    slope is below FLAT_SLOPE_DEG.
    """
    width_m, height_m = cell_sizes_m(transform, dem.shape[0])
    z = np.pad(dem, 1, mode="reflect", reflect_type="odd")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    rise_east = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * width_m[:, None])
    rise_north = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * height_m[:, None])
    slope = np.degrees(np.arctan(np.hypot(rise_east, rise_north)))
    # The downhill direction is the negative gradient; bearing is measured from north.
    aspect = np.degrees(np.arctan2(-rise_east, -rise_north)) % 360.0
    slope[np.isnan(dem)] = np.nan  # a cell without a height has no slope of its own
    aspect[(slope < FLAT_SLOPE_DEG) | np.isnan(slope)] = np.nan
    return slope, aspect


def fill_depressions(dem: Grid) -> Grid:
    """Priority-flood depression filling with an epsilon (Barnes et al. 2014).

    Every pit is raised to its spill level and every cell of a filled flat is nudged up by the
    smallest float step per cell away from the outlet, so each cell has a strictly lower
    neighbour on the way out. Cells on the array edge, and next to NaN, are the outlets. The
    result is for routing water only; slope and aspect use the original heights.
    """
    rows, cols = dem.shape
    filled = dem.copy()
    seen = np.isnan(dem)
    heap: list[tuple[float, int]] = []
    nan = np.isnan(dem)
    for r in range(rows):
        for c in range(cols):
            if nan[r, c]:
                continue
            edge = r in (0, rows - 1) or c in (0, cols - 1)
            if edge or _touches_nodata(nan, r, c):
                heap.append((float(dem[r, c]), r * cols + c))
                seen[r, c] = True
    heapq.heapify(heap)
    while heap:
        height, index = heapq.heappop(heap)
        r, c = divmod(index, cols)
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                seen[nr, nc] = True
                level = max(float(dem[nr, nc]), math.nextafter(height, math.inf))
                filled[nr, nc] = level
                heapq.heappush(heap, (level, nr * cols + nc))
    return filled


def _touches_nodata(nan: NDArray[np.bool_], r: int, c: int) -> bool:
    return bool(nan[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2].any())


def _receivers(filled: Grid, transform: Affine) -> NDArray[np.int64]:
    """Flat index of the steepest-descent neighbour of each cell, -1 where there is none."""
    rows, cols = filled.shape
    width_m, height_m = cell_sizes_m(transform, rows)
    z = np.pad(filled, 1, constant_values=np.nan)
    best = np.zeros(filled.shape)
    receiver = np.full(filled.shape, -1, dtype=np.int64)
    flat = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    for dr, dc in NEIGHBOURS:
        neighbour = z[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        run = np.hypot(width_m * abs(dc), height_m * abs(dr))[:, None]
        with np.errstate(invalid="ignore"):
            gradient = (filled - neighbour) / run
        better = gradient > best  # NaN compares False: nodata is never a receiver
        best = np.where(better, gradient, best)
        target = np.pad(flat, 1, constant_values=-1)[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        receiver = np.where(better, target, receiver)
    receiver[np.isnan(filled)] = -1
    return receiver


def outlet_mask(dem: Grid, transform: Affine) -> NDArray[np.bool_]:
    """Valid cells that pass their water on to no other cell (they hold a whole catchment)."""
    receiver = _receivers(fill_depressions(dem), transform)
    mask: NDArray[np.bool_] = (receiver == -1) & ~np.isnan(dem)
    return mask


def flow_accumulation(dem: Grid, transform: Affine) -> Grid:
    """D8 flow accumulation: the number of cells draining through each cell, itself included.

    Depressions ARE filled first (priority flood with epsilon, see fill_depressions), so water
    never stops in a pit and every cell reaches an outlet at the array edge or at nodata. Flow
    goes to the steepest of the eight neighbours, with distances in metres. D8 sends all water
    to one neighbour, so it makes thin, single-cell channels on smooth slopes. Only cells inside
    the array are counted: catchments that extend beyond it are truncated.
    """
    filled = fill_depressions(dem)
    receiver = _receivers(filled, transform).ravel()
    height = filled.ravel()
    valid = np.flatnonzero(~np.isnan(height))
    order = valid[np.argsort(-height[valid], kind="stable")]  # highest first
    accumulation = np.where(np.isnan(height), 0, 1).tolist()
    target = receiver.tolist()
    for index in order.tolist():
        downstream = target[index]
        if downstream >= 0:
            accumulation[downstream] += accumulation[index]
    result = np.asarray(accumulation, dtype="float64").reshape(dem.shape)
    result[np.isnan(dem)] = np.nan
    return result
