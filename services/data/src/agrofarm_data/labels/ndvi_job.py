"""Build the `ndvi_cell_year` table for the experimental NDVI model.

    python -m agrofarm_data.labels.ndvi_job download --out <dir> [--start-year 2000 --end-year 2025]
    python -m agrofarm_data.labels.ndvi_job build --modis <dir> --worldcover <dir> --chirps <dir> \
        --power <json> --enso <parquet> --out <dir>

`download` reads MODIS MOD13Q1 NDVI (Terra, tile h09v07) from the Planetary Computer for the
composites that start in May to October. It reads the overview at a quarter of the resolution
(about 1 km) because the full 250 m band takes about a minute per composite on a home line.

Limits of that choice, stated in the manifest too: the overview is a resampled picture, so the
pixel reliability band cannot be applied to it. Cloudy dates are handled only by taking the
seasonal maximum, and pixels next to water or to missing data can be biased low.

`build` needs, on disk: the composites, the ESA WorldCover 2021 tiles, the CHIRPS v3 monthly
files, one NASA POWER monthly regional JSON for T2M, and the clean `enso_monthly` table.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from agrofarm_data.connectors import planetary_computer as pc
from agrofarm_data.labels.ndvi_peak import (
    cropland_weighted_cells,
    sample_sinusoidal,
    seasonal_peak,
    sinusoidal_xy,
)
from agrofarm_data.labels.ndvi_table import SEASON, CellGrid, TableInputs, build_cell_year_table
from agrofarm_data.manifest import validate_manifest

DATASET = "ndvi_cell_year"
PROCESSING_VERSION = "ndvi-cell-year-0.1.0"
BBOX = (-92.5, 13.5, -88.0, 18.0)  # west, south, east, north: Guatemala and its neighbours
CELL = 0.05  # degrees, the CHIRPS grid
FINE = 0.01  # degrees, the grid the NDVI and the cropland map meet on
FACTOR = round(CELL / FINE)
OVERVIEW = 4  # read MODIS at a quarter of the 250 m resolution
TILE = "h09v07"
ASSET = "250m_16_days_NDVI"
CROPLAND_CLASS = 40
WORLDCOVER_PIXELS_PER_FINE = 120  # 10 m pixels (1/12000 degree) per 0.01 degree
MIN_COMPOSITES = 6
MIN_COVER = 0.8
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_TIMEOUT": "120",
}
Array = NDArray[Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- download ---------------------------------------------------------------------------------


def _read_overview(href: str) -> tuple[Array, tuple[float, ...]]:
    with rasterio.open(href) as source:
        bounds = transform_bounds("EPSG:4326", source.crs, *BBOX, 21)
        window = from_bounds(*bounds, source.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, source.width, source.height))
        shape = (int(window.height // OVERVIEW), int(window.width // OVERVIEW))
        data = source.read(1, window=window, out_shape=shape)
        transform = source.window_transform(window)
        transform = transform * transform.scale(window.width / shape[1], window.height / shape[0])
        return data, tuple(transform)[:6]


def download(out_dir: Path, start_year: int, end_year: int, workers: int = 3) -> int:
    """Write one `.npz` per composite. Files already there are kept, so a rerun resumes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = pc.SPECS["modis-13q1-061"]
    items: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        start, end = f"{year}-{SEASON[0]:02d}-01", f"{year}-{SEASON[1]:02d}-31"
        found = pc.search_items(pc.http_json, spec, BBOX, start, end, 100.0)
        items += [x for x in found if x["id"].startswith("MOD13Q1") and TILE in x["id"]]
    token = [pc.fetch_token(pc.http_json, spec)]
    fetched_at = [time.monotonic()]

    def signed(href: str) -> str:
        if time.monotonic() - fetched_at[0] > 1500:  # the token lasts about an hour
            token[0], fetched_at[0] = pc.fetch_token(pc.http_json, spec), time.monotonic()
        return f"{href}?{token[0]}"

    def one(item: dict[str, Any]) -> str | None:
        target = out_dir / f"{item['id']}.npz"
        if target.exists():
            return None
        href = item["assets"][ASSET]["href"]
        error = ""
        for attempt in range(4):
            try:
                data, transform = _read_overview(signed(href))
            except (rasterio.errors.RasterioError, OSError) as exc:
                error = repr(exc)[:200]
                time.sleep(5.0 * (attempt + 1))
                continue
            partial = target.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                ndvi=data,
                transform=np.array(transform),
                start=item["properties"]["start_datetime"][:10],
                href=href,
            )
            partial.rename(target)
            return None
        return f"{item['id']}: {error}"

    with rasterio.Env(**GDAL_ENV), ThreadPoolExecutor(workers) as pool:
        failures = [message for message in pool.map(one, items) if message]
    for message in failures:
        sys.stderr.write(f"FAILED {message}\n")
    sys.stdout.write(f"{len(items) - len(failures)} of {len(items)} composites in {out_dir}\n")
    return 1 if failures else 0


# --- inputs on disk ---------------------------------------------------------------------------


def cropland_fine_grid(worldcover_dir: Path) -> Array:
    """Cropland fraction (class 40) on the 0.01 degree grid. NaN where no tile has data."""
    west, south, east, north = BBOX
    rows, cols = round((north - south) / FINE), round((east - west) / FINE)
    fraction = np.full((rows, cols), np.nan, dtype="float32")
    block = WORLDCOVER_PIXELS_PER_FINE
    for path in sorted(worldcover_dir.glob("*_Map.tif")):
        with rasterio.open(path) as source:
            bounds = source.bounds
            w, e = max(bounds.left, west), min(bounds.right, east)
            s, n = max(bounds.bottom, south), min(bounds.top, north)
            if w >= e or s >= n:
                continue
            window = from_bounds(w, s, e, n, source.transform).round_offsets().round_lengths()
            height, width = int(window.height), int(window.width)
            row0, col0 = round((north - n) / FINE), round((w - west) / FINE)
            strip = block * 10
            for offset in range(0, height, strip):
                size = min(strip, height - offset)
                data = source.read(
                    1, window=Window(window.col_off, window.row_off + offset, width, size)
                )
                shape = (size // block, block, width // block, block)
                mapped = (data != 0).reshape(shape).sum(axis=(1, 3))
                crops = (data == CROPLAND_CLASS).reshape(shape).sum(axis=(1, 3))
                target = (
                    slice(row0 + offset // block, row0 + (offset + size) // block),
                    slice(col0, col0 + width // block),
                )
                fraction[target] = np.where(mapped > 0, crops / np.maximum(mapped, 1), np.nan)
    return fraction


def chirps_monthly(chirps_dir: Path) -> tuple[Array, Array]:
    """Monthly rainfall (mm) on the 0.05 degree grid of the box, and YYYYMM of each layer."""
    months, layers = [], []
    for path in sorted(chirps_dir.glob("chirps-v3.0.*.tif")):
        match = re.search(r"(\d{4})\.(\d{2})\.tif$", path.name)
        if match is None:
            continue
        with rasterio.open(path) as source:
            window = from_bounds(*BBOX, source.transform).round_offsets().round_lengths()
            data = source.read(1, window=window).astype("float32")
            if source.nodata is not None:
                data[data == source.nodata] = np.nan
        data[data < 0] = np.nan  # negative rain is a missing-value marker, never rain
        months.append(int(match.group(1)) * 100 + int(match.group(2)))
        layers.append(data)
    return np.stack(layers), np.array(months)


def power_season_temperature(power_json: Path, grid: CellGrid) -> dict[int, Array]:
    """Season mean of T2M per cell, from the nearest NASA POWER grid point (about 50 km)."""
    payload = json.loads(power_json.read_text(encoding="utf-8"))
    fill = float(payload["header"]["fill_value"])
    points = [
        (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"])
        for f in payload["features"]
    ]
    lon, lat = np.meshgrid(grid.longitudes(), grid.latitudes())
    point_lon = np.array([p[0] for p in points])
    point_lat = np.array([p[1] for p in points])
    distance = (lon[..., None] - point_lon) ** 2 + (lat[..., None] - point_lat) ** 2
    nearest = distance.argmin(axis=-1)
    season = [f"{month:02d}" for month in range(SEASON[0], SEASON[1] + 1)]
    years = sorted({int(key[:4]) for key in points[0][2]["parameter"]["T2M"]})
    result: dict[int, Array] = {}
    for year in years:
        values = []
        for _, _, properties in points:
            series = properties["parameter"]["T2M"]
            months = [series.get(f"{year}{month}", fill) for month in season]
            values.append(np.nan if fill in months else float(np.mean(months)))
        result[year] = np.array(values)[nearest]
    return result


def seasonal_peaks(modis_dir: Path, cropland: Array) -> tuple[dict[int, Array], dict[int, int]]:
    """Cropland-weighted seasonal peak NDVI per 0.05 degree cell and year."""
    west, _, _, north = BBOX
    rows, cols = cropland.shape
    lat = north - (np.arange(rows) + 0.5) * FINE
    lon = west + (np.arange(cols) + 0.5) * FINE
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = sinusoidal_xy(lat_grid, lon_grid)
    by_year: dict[int, list[Array]] = {}
    for path in sorted(modis_dir.glob("MOD13Q1*.npz")):
        with np.load(path) as composite:
            start = str(composite["start"])
            if not SEASON[0] <= int(start[5:7]) <= SEASON[1]:
                continue
            sampled = sample_sinusoidal(composite["ndvi"], composite["transform"], x, y)
        by_year.setdefault(int(start[:4]), []).append(sampled.astype("float32"))
    peaks, counts = {}, {}
    for year, stack in sorted(by_year.items()):
        peak = seasonal_peak(np.stack(stack), MIN_COMPOSITES)
        peaks[year], _ = cropland_weighted_cells(peak, cropland, FACTOR, MIN_COVER)
        counts[year] = len(stack)
    return peaks, counts


# --- build ------------------------------------------------------------------------------------


def build(
    modis_dir: Path,
    worldcover_dir: Path,
    chirps_dir: Path,
    power_json: Path,
    enso_path: Path,
    out_dir: Path,
) -> Path:
    west, south, east, north = BBOX
    grid = CellGrid(west, north, CELL, round((north - south) / CELL), round((east - west) / CELL))
    cropland = cropland_fine_grid(worldcover_dir)
    peaks, counts = seasonal_peaks(modis_dir, cropland)
    _, cropland_cells = cropland_weighted_cells(
        np.zeros_like(cropland, dtype="float64"), cropland, FACTOR, MIN_COVER
    )
    rain, rain_months = chirps_monthly(chirps_dir)
    table = build_cell_year_table(
        grid,
        TableInputs(
            peaks=peaks,
            cropland=cropland_cells,
            rain=rain,
            rain_months=rain_months,
            temperature=power_season_temperature(power_json, grid),
            enso=pd.read_parquet(enso_path),
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{DATASET}.parquet"
    table.to_parquet(target, index=False)
    composites = sorted(modis_dir.glob("MOD13Q1*.npz"))
    manifest = {
        "dataset": DATASET,
        "file": target.name,
        "source": "MODIS MOD13Q1 v061 NDVI (Planetary Computer), ESA WorldCover 2021 v200, "
        "CHIRPS v3 monthly, NASA POWER monthly T2M (MERRA-2), NOAA CPC RONI",
        "retrieved_at": datetime.now(UTC).date().isoformat(),
        "license": "Mixed, one per source. See the manifest of each source dataset. "
        "License wording of the MODIS collection on the Planetary Computer is not verified.",
        "attribution": "NASA LP DAAC MOD13Q1; ESA WorldCover project 2021 (CC BY 4.0); "
        "CHIRPS, Climate Hazards Center UC Santa Barbara; NASA POWER; NOAA CPC.",
        "date_range": {
            "start": f"{int(table['year'].min())}-{SEASON[0]:02d}-01",
            "end": f"{int(table['year'].max())}-{SEASON[1]:02d}-31",
        },
        "time_step": "one row per cell and rainy season (May to October)",
        "spatial_resolution": f"{CELL} degrees (about 5.5 km); NDVI read at about 1 km",
        "checksum_sha256": _sha256(target),
        "source_checksum_sha256": {path.name: _sha256(path) for path in composites}
        | {power_json.name: _sha256(power_json), enso_path.name: _sha256(enso_path)},
        "processing_version": PROCESSING_VERSION,
        "bbox": list(BBOX),
        "rows": int(len(table)),
        "cells": int(table["cell_id"].nunique()),
        "composites_per_year": {str(year): count for year, count in counts.items()},
        "label": "ndvi_peak: maximum NDVI of the season, averaged over the cell with cropland "
        "fraction as weight. A proxy for crop condition. It is not yield.",
        "limitations": [
            "NDVI overview at a quarter resolution: the pixel reliability band is not applied.",
            "Clouds are handled only by the seasonal maximum.",
            "The cropland map is from 2021 and is used for every year.",
            "Temperature comes from a reanalysis grid of about 50 km, the nearest point.",
            "Cells with less than 5 % cropland, or with any missing input, have no row.",
        ],
    }
    validate_manifest(manifest)
    (out_dir / f"{DATASET}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrofarm_data.labels.ndvi_job")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--start-year", type=int, default=2000)
    fetch.add_argument("--end-year", type=int, default=2025)
    make = commands.add_parser("build")
    for name in ("modis", "worldcover", "chirps", "power", "enso", "out"):
        make.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        return download(args.out, args.start_year, args.end_year)
    target = build(args.modis, args.worldcover, args.chirps, args.power, args.enso, args.out)
    sys.stdout.write(f"{target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
