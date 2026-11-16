"""Read Copernicus GLO-30 DEM tiles and mosaic the part that covers a box.

GLO-30 is distributed as one GeoTIFF per degree (3600 x 3600 cells of 1 arc second, about 30 m
at the equator), EPSG:4326, heights in metres above the EGM2008 geoid. It is a surface model
(DSM). The tiles in the team download carry no nodata flag, so a declared nodata value is
honoured when present and non-finite or implausibly low heights are also read as no data.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds

Bbox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)

# Extra ground read around the box so slope at the box edge and the upslope area of cells near
# it are computed from real neighbours. 0.02 degrees is about 2 km.
MARGIN_DEG = 0.02
# Heights at or below this are not terrain (the lowest dry land on Earth is about -430 m).
MIN_PLAUSIBLE_HEIGHT_M = -500.0
GRID_TOLERANCE = 1e-3  # tiles must sit on the same cell grid within this share of a cell


class TerrainError(ValueError):
    """No usable DEM tile, or tiles that do not line up."""


@dataclass(frozen=True)
class Dem:
    """Heights in metres (NaN for no data) on a grid, plus where the box sits inside it."""

    values: NDArray[np.float64]
    transform: Affine
    crop: Window  # the cells inside the requested box
    tiles: list[Path]


def _overlapping(raw_dir: Path, bbox: Bbox, margin: float) -> list[Path]:
    west, south, east, north = bbox
    found = []
    for path in sorted(raw_dir.glob("*.tif")):
        with rasterio.open(path) as source:
            left, bottom, right, top = source.bounds
        inside_lon = left < east + margin and right > west - margin
        inside_lat = bottom < north + margin and top > south - margin
        if inside_lon and inside_lat:
            found.append(path)
    return found


def _tile_offset(
    source: rasterio.DatasetReader, origin: tuple[float, float], size: tuple[float, float]
) -> tuple[int, int]:
    """Whole-cell offset of a tile on the shared grid, or an error if it is off the grid."""
    name = Path(source.name).name
    if abs(source.res[0] - size[0]) > size[0] * GRID_TOLERANCE:
        raise TerrainError(f"{name} has a different cell size: not on the same grid")
    col = (source.bounds.left - origin[0]) / size[0]
    row = (origin[1] - source.bounds.top) / size[1]
    if abs(col - round(col)) > GRID_TOLERANCE or abs(row - round(row)) > GRID_TOLERANCE:
        raise TerrainError(f"{name} is not aligned to the cell grid of the other tiles")
    return round(col), round(row)


def read_dem(raw_dir: Path, bbox: Bbox, margin_deg: float = MARGIN_DEG) -> Dem:
    """Mosaic the tiles that touch the box (plus a margin) onto one grid.

    The margin is clipped to the ground the tiles cover. Where a tile is missing inside the
    mosaic the cells are NaN.
    """
    west, south, east, north = bbox
    if not _overlapping(raw_dir, bbox, 0.0):
        raise TerrainError(f"No DEM tile in {raw_dir} covers the box {bbox}")
    paths = _overlapping(raw_dir, bbox, margin_deg)
    sources = [rasterio.open(path) for path in paths]
    try:
        first = sources[0]
        size = (first.res[0], first.res[1])
        origin = (first.bounds.left, first.bounds.top)
        offsets = [_tile_offset(source, origin, size) for source in sources]
        extents = list(zip(offsets, sources, strict=True))
        # Wanted area in grid cells, clipped to what the tiles cover.
        col0 = max(
            math.floor((west - margin_deg - origin[0]) / size[0]), min(o[0] for o in offsets)
        )
        row0 = max(
            math.floor((origin[1] - (north + margin_deg)) / size[1]), min(o[1] for o in offsets)
        )
        col1 = min(
            math.ceil((east + margin_deg - origin[0]) / size[0]),
            max(o[0] + s.width for o, s in extents),
        )
        row1 = min(
            math.ceil((origin[1] - (south - margin_deg)) / size[1]),
            max(o[1] + s.height for o, s in extents),
        )
        values = np.full((row1 - row0, col1 - col0), np.nan, dtype="float64")
        for (off_col, off_row), source in extents:
            c_start, c_end = max(col0, off_col), min(col1, off_col + source.width)
            r_start, r_end = max(row0, off_row), min(row1, off_row + source.height)
            if c_start >= c_end or r_start >= r_end:
                continue
            window = Window(c_start - off_col, r_start - off_row, c_end - c_start, r_end - r_start)
            block = source.read(1, window=window).astype("float64")
            bad = ~np.isfinite(block) | (block <= MIN_PLAUSIBLE_HEIGHT_M)
            if source.nodata is not None and np.isfinite(source.nodata):
                bad |= block == source.nodata
            block[bad] = np.nan
            target = values[r_start - row0 : r_end - row0, c_start - col0 : c_end - col0]
            np.copyto(target, block, where=~np.isnan(block))
    finally:
        for source in sources:
            source.close()
    transform = Affine(
        size[0], 0.0, origin[0] + col0 * size[0], 0.0, -size[1], origin[1] - row0 * size[1]
    )
    wanted = from_bounds(west, south, east, north, transform)
    crop = wanted.round_offsets().round_lengths()
    crop = crop.intersection(Window(0, 0, values.shape[1], values.shape[0]))
    return Dem(values=values, transform=transform, crop=crop, tiles=paths)
