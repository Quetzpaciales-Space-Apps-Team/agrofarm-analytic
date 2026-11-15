"""ESA WorldCover 2021 v200 (10 m land cover) as a cropland mask on the 0.05 degree grid.

The grid is the CHIRPS grid: cells are aligned to multiples of 0.05 degrees in EPSG:4326. For
every cell the connector counts the 10 m pixels of each class. The pixels are counted, never
resampled, so a fraction is exactly the share of valid pixels in that class. Pixels are assigned
to the cell that holds their centre. Tiles (3 x 3 degrees) are read in windows, block by block,
and mosaicked by the grid: a box that crosses a tile edge needs no special case.

No data: the team files use the flag 0 (also the value the product documents as "no data"). Both
the flag of the file, if any, and 0 are no data. They leave the denominator, so a cell where
every pixel is no data gets null fractions, never zero.

The cropland mask is a screening aid for training cells, not a map of what is grown.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window

from agrofarm_data import __version__

DATASET = "landcover_worldcover_cropland_mask"
SOURCE_YEAR = 2021
SOURCE_VERSION = "v200"
ATTRIBUTION = (
    "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) "
    "processed by ESA WorldCover consortium"
)
Bbox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)

CELL_DEGREES = 0.05  # the CHIRPS grid
CROPLAND_CLASS = 40
# Team assumption, to be reviewed: a cell is cropland when at least half of its valid 10 m
# pixels are class 40, and at least half of the cell has valid pixels. Both are inclusive.
CROPLAND_THRESHOLD = 0.5
MIN_VALID_FRACTION = 0.5

CLASSES: dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
_CODES = tuple(CLASSES)
_NODATA_BUCKET = len(_CODES)  # bucket for no data pixels
_UNKNOWN_BUCKET = _NODATA_BUCKET + 1  # bucket for values that are not a WorldCover class
_BUCKETS = _NODATA_BUCKET + 1  # counted buckets: the classes and no data
_CHUNK = 1024  # target window side in pixels (at most 2048 for wide blocks)
LIMITATIONS = [
    "The cropland class does not say which crop is grown, nor whether the field is in use "
    "or fallow.",
    "One year only (2021, v200). Land use changed since then is not seen.",
    "The classification of 10 m pixels has errors (the producer reports about 75 percent "
    "overall accuracy at global scale). Check the farm on a recent image.",
    "Built-up pixels are a separate class. Fields that are small, narrow or mixed with trees "
    "or houses can be labelled as other classes, and the fraction under-counts them.",
    "The mask threshold is a team assumption to be reviewed, not a validated value.",
    "A pixel value of 0 (file nodata flag) is no data. Check valid_fraction before using "
    "a fraction; a cell without valid pixels has null fractions.",
]


class WorldCoverError(ValueError):
    """The tiles are missing, do not cover the box or hold values that are not WorldCover."""


@dataclass(frozen=True)
class Grid:
    """Cells of 0.05 degrees. Cell column i spans lon [(west_index + i) * 0.05, ... + 0.05)."""

    west_index: int
    south_index: int
    nx: int
    ny: int

    @property
    def west(self) -> float:
        return round(self.west_index * CELL_DEGREES, 6)

    @property
    def east(self) -> float:
        return round((self.west_index + self.nx) * CELL_DEGREES, 6)

    @property
    def south(self) -> float:
        return round(self.south_index * CELL_DEGREES, 6)

    @property
    def north(self) -> float:
        return round((self.south_index + self.ny) * CELL_DEGREES, 6)

    @property
    def transform(self) -> Affine:
        return from_origin(self.west, self.north, CELL_DEGREES, CELL_DEGREES)


def grid_for(bbox: Bbox) -> Grid:
    """The smallest set of 0.05 degree cells that covers the box."""
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise WorldCoverError(f"Not a valid box (west south east north in EPSG:4326): {bbox}")
    # round() removes float noise such as -90.65 / 0.05 = -1812.9999999999998
    i0, i1 = (
        math.floor(round(west / CELL_DEGREES, 6)),
        math.ceil(round(east / CELL_DEGREES, 6)),
    )
    j0, j1 = (
        math.floor(round(south / CELL_DEGREES, 6)),
        math.ceil(round(north / CELL_DEGREES, 6)),
    )
    return Grid(west_index=i0, south_index=j0, nx=i1 - i0, ny=j1 - j0)


def _column(name: str) -> str:
    return f"fraction_{name}"


def select_tiles(raw_dir: Path, grid: Grid) -> list[Path]:
    """The tiles of the folder that overlap the grid. Raises if there are none."""
    found = sorted(raw_dir.glob("*_Map.tif"))
    if not found:
        raise WorldCoverError(f"No WorldCover tiles (*_Map.tif) in {raw_dir}")
    chosen: list[Path] = []
    for path in found:
        with rasterio.open(path) as source:
            left, bottom, right, top = source.bounds
        if grid.west < right and grid.east > left and grid.south < top and grid.north > bottom:
            chosen.append(path)
    if not chosen:
        raise WorldCoverError(
            f"The box {grid.west} {grid.south} {grid.east} {grid.north} is outside every tile "
            f"in {raw_dir}"
        )
    return chosen


def _lookup(nodata: float | None) -> NDArray[np.uint8]:
    lut = np.full(256, _UNKNOWN_BUCKET, dtype="uint8")
    for bucket, code in enumerate(_CODES):
        lut[code] = bucket
    lut[0] = _NODATA_BUCKET
    if nodata is not None and 0 <= nodata <= 255 and float(nodata).is_integer():
        lut[int(nodata)] = _NODATA_BUCKET
    return lut


def _pixel_range(origin: float, size: float, low: float, high: float, count: int) -> range:
    """Pixels whose centre lies in [low, high), for an axis that starts at `origin`."""
    start = math.ceil(round((low - origin) / size - 0.5, 6))
    stop = math.ceil(round((high - origin) / size - 0.5, 6))
    return range(max(start, 0), min(stop, count))


def _steps(block: int, count: int) -> int:
    if block > 2 * _CHUNK:
        return 2 * _CHUNK
    return min(max(block, (_CHUNK // block) * block), max(count, 1))


def _count_tile(path: Path, grid: Grid, counts: NDArray[np.int64]) -> tuple[float, float]:
    """Add the class counts of one tile to `counts` (ny, nx, buckets). Returns the pixel size."""
    with rasterio.open(path) as source:
        transform = source.transform
        px, py = transform.a, -transform.e
        # latitude decreases with the row: rows between north (small row) and south (large row)
        rows = range(
            max(math.ceil(round((transform.f - grid.north) / py - 0.5, 6)), 0),
            min(math.ceil(round((transform.f - grid.south) / py - 0.5, 6)), source.height),
        )
        cols = _pixel_range(transform.c, px, grid.west, grid.east, source.width)
        if not rows or not cols:
            return px, py
        lut = _lookup(source.nodata)
        block_rows, block_cols = source.block_shapes[0]
        step_r = _steps(block_rows, source.height)
        step_c = _steps(block_cols, source.width)
        for r0 in range(rows.start - rows.start % step_r, rows.stop, step_r):
            r_lo, r_hi = max(r0, rows.start), min(r0 + step_r, rows.stop)
            lat = transform.f - (np.arange(r_lo, r_hi) + 0.5) * py
            row_cell = (grid.south_index + grid.ny - 1) - np.floor(lat / CELL_DEGREES).astype(
                np.int64
            )
            for c0 in range(cols.start - cols.start % step_c, cols.stop, step_c):
                c_lo, c_hi = max(c0, cols.start), min(c0 + step_c, cols.stop)
                lon = transform.c + (np.arange(c_lo, c_hi) + 0.5) * px
                col_cell = np.floor(lon / CELL_DEGREES).astype(np.int64) - grid.west_index
                window = Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
                values = source.read(1, window=window)
                bucket = lut[values]
                unknown = bucket == _UNKNOWN_BUCKET
                if unknown.any():
                    bad = np.unique(values[unknown]).tolist()
                    raise WorldCoverError(f"{path.name}: not a land cover class: {bad}")
                key = (row_cell[:, None] * grid.nx + col_cell[None, :]) * _BUCKETS + bucket
                counts += np.bincount(key.ravel(), minlength=counts.size).reshape(counts.shape)
    return px, py


def aggregate(raw_dir: Path, bbox: Bbox) -> pd.DataFrame:
    """One row per 0.05 degree cell (north to south, west to east) with class fractions."""
    grid = grid_for(bbox)
    tiles = select_tiles(raw_dir, grid)
    sizes: set[tuple[float, float]] = set()
    counts = np.zeros((grid.ny, grid.nx, _BUCKETS), dtype=np.int64)
    for path in tiles:
        px, py = _count_tile(path, grid, counts)
        sizes.add((round(px, 12), round(py, 12)))
    if len(sizes) > 1:
        raise WorldCoverError(f"The tiles have different pixel size: {sorted(sizes)}")
    ((px, py),) = sizes
    per_cell = round(CELL_DEGREES / px) * round(CELL_DEGREES / py)

    classes = counts[:, :, :_NODATA_BUCKET].astype("float64")
    valid = counts[:, :, :_NODATA_BUCKET].sum(axis=2)
    if (valid + counts[:, :, _NODATA_BUCKET] > per_cell).any():
        raise WorldCoverError("More pixels than a cell holds: overlapping tiles or odd pixel size")
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = np.where(valid[:, :, None] > 0, classes / valid[:, :, None], np.nan)
    valid_fraction = valid / per_cell

    lon = grid.west + (np.arange(grid.nx) + 0.5) * CELL_DEGREES
    lat = grid.north - (np.arange(grid.ny) + 0.5) * CELL_DEGREES
    table = pd.DataFrame(
        {
            "lon": np.tile(lon, grid.ny).round(6),
            "lat": np.repeat(lat, grid.nx).round(6),
        }
    )
    for bucket, name in enumerate(CLASSES.values()):
        table[_column(name)] = fractions[:, :, bucket].ravel()
    table["valid_pixels"] = valid.ravel()
    table["valid_fraction"] = valid_fraction.ravel().round(6)
    cropland = table[_column(CLASSES[CROPLAND_CLASS])]
    table["cropland_mask"] = (
        (cropland >= CROPLAND_THRESHOLD) & (table["valid_fraction"] >= MIN_VALID_FRACTION)
    ).astype(bool)
    return table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_raster(path: Path, values: NDArray[np.float32], grid: Grid) -> None:
    profile = {
        "driver": "GTiff", "height": grid.ny, "width": grid.nx, "count": 1, "dtype": "float32",
        "crs": "EPSG:4326", "transform": grid.transform, "nodata": float("nan"),
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as out:
        out.write(values, 1)


def write_clean(raw_dir: Path, out_dir: Path, bbox: Bbox, area_name: str) -> Path:
    """Write the table, the cropland fraction GeoTIFF and the manifest. Returns the manifest."""
    grid = grid_for(bbox)
    table = aggregate(raw_dir, bbox)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{DATASET}.parquet"
    table.to_parquet(table_path, index=False, engine="pyarrow")
    raster_path = out_dir / "cropland_fraction.tif"
    cropland_column = _column(CLASSES[CROPLAND_CLASS])
    _write_raster(
        raster_path, table[cropland_column].to_numpy("float32").reshape(grid.ny, grid.nx), grid
    )

    pixels = table["valid_pixels"].to_numpy("float64")
    total = float(pixels.sum())
    shares = {
        name: round(float(np.nansum(table[_column(name)].to_numpy() * pixels) / total), 6)
        if total
        else None
        for name in CLASSES.values()
    }
    west, south, east, north = bbox
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "ESA WorldCover 2021 v200 (10 m), https://esa-worldcover.org",
        "source_year": SOURCE_YEAR,
        "source_version": SOURCE_VERSION,
        "license": "CC BY 4.0",
        "attribution": ATTRIBUTION,
        "area_name": area_name,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "grid_extent": {
            "west": grid.west,
            "south": grid.south,
            "east": grid.east,
            "north": grid.north,
        },  # fmt: skip
        "grid": {
            "cell_degrees": CELL_DEGREES,
            "crs": "EPSG:4326",
            "alignment": "cell edges at multiples of 0.05 degrees (the CHIRPS grid)",
            "cell_value": "column lon and lat are the cell centre",
        },
        "method": "pixel counts per cell by pixel centre, no resampling; "
        "no data pixels leave the denominator",
        "classes": {str(code): name for code, name in CLASSES.items()},
        "threshold": {
            "cropland_class": CROPLAND_CLASS,
            "cropland_fraction": CROPLAND_THRESHOLD,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "rule": "cropland_mask = fraction_cropland >= cropland_fraction "
            "and valid_fraction >= min_valid_fraction",
            "status": "team assumption, to be reviewed",
        },
        "rows": int(len(table)),
        "cropland_cells": int(table["cropland_mask"].sum()),
        "cells_without_data": int((table["valid_pixels"] == 0).sum()),
        "class_share_of_valid_pixels": shares,
        "time_step": "static (2021 classification)",
        "spatial_resolution": "10 m source, aggregated to 0.05 degree cells",
        "missing_value": "null fraction when the cell has no valid pixel",
        "limitations": LIMITATIONS,
        "tiles": [
            {"file": path.name, "checksum_sha256": _sha256(path)}
            for path in select_tiles(raw_dir, grid)
        ],
        "raster": {
            "file": raster_path.name,
            "content": "cropland fraction, float32, nodata NaN",
            "checksum_sha256": _sha256(raster_path),
        },
        "checksum_sha256": _sha256(table_path),
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
