"""CHIRPS v3 final daily RNL rainfall on the native 0.05 degree training grid.

The area is supplied by the caller until DEC-003 publishes the pilot training polygon.
Daily RNL amounts partition pentadal totals with ERA5 daily precipitation; they are
unsuitable for estimating one-day rainfall extremes.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin
from rasterio.windows import Window

from agrofarm_data import __version__

DATASET = "climate_chirps_daily"
VERSION = "v3.0-final-rnl"
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl"
CELL_DEGREES = 0.05
SOURCE_NODATA = -9999.0
OUTPUT_NODATA = -9999.0
LIMITATIONS = [
    "Daily values are disaggregated from pentads using ERA5 rainfall and are not valid for "
    "one-day extremes. Date labels do not represent a uniform local observation window."
]
_NAME = re.compile(r"^chirps-v3\.0\.rnl\.(\d{4})\.(\d{2})\.(\d{2})\.tif$")
Bbox = tuple[float, float, float, float]  # west, south, east, north, EPSG:4326
WindowReader = Callable[[str, Bbox], bytes]


class ChirpsError(ValueError):
    """The requested area or a CHIRPS source file is invalid."""


def _read_remote_window(url: str, bbox: Bbox) -> bytes:
    """Read only the requested pixels through GDAL's HTTP range support."""
    west, north, width, height = _grid(bbox)
    try:
        with rasterio.open(f"/vsicurl/{url}") as source:
            transform = source.transform
            if source.crs is None or source.crs.to_epsg() != 4326 or source.count != 1:
                raise ChirpsError("CHIRPS response must contain one EPSG:4326 raster band")
            if (
                not math.isclose(transform.a, CELL_DEGREES, abs_tol=1e-7)
                or not math.isclose(-transform.e, CELL_DEGREES, abs_tol=1e-7)
                or not math.isclose(transform.b, 0, abs_tol=1e-9)
                or not math.isclose(transform.d, 0, abs_tol=1e-9)
            ):
                raise ChirpsError("CHIRPS response is not an unrotated 0.05 degree grid")
            col = round((west - transform.c) / CELL_DEGREES)
            row = round((transform.f - north) / CELL_DEGREES)
            if (
                abs(transform.c + col * CELL_DEGREES - west) > 1e-5
                or abs(transform.f - row * CELL_DEGREES - north) > 1e-5
            ):
                raise ChirpsError("CHIRPS response is not aligned to the requested grid")
            if col < 0 or row < 0 or col + width > source.width or row + height > source.height:
                raise ChirpsError("Training area is outside CHIRPS source coverage")
            values = source.read(1, window=Window(col, row, width, height))
            nodata = source.nodata if source.nodata is not None else SOURCE_NODATA
            profile = {
                "driver": "GTiff", "width": width, "height": height, "count": 1,
                "dtype": "float32", "crs": "EPSG:4326",
                "transform": from_origin(west, north, CELL_DEGREES, CELL_DEGREES),
                "nodata": nodata,
            }  # fmt: skip
            with MemoryFile() as memory:
                with memory.open(**profile) as output:
                    output.write(values.astype("float32"), 1)
                    output.update_tags(source_url=url)
                return bytes(memory.read())
    except rasterio.errors.RasterioIOError as exc:
        raise ChirpsError("Unable to read the CHIRPS HTTP raster window") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid(bbox: Bbox) -> tuple[float, float, int, int]:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -60 <= south < north <= 60):
        raise ChirpsError("Invalid training-area bounding box in EPSG:4326")
    left = math.floor(round(west / CELL_DEGREES, 6))
    right = math.ceil(round(east / CELL_DEGREES, 6))
    bottom = math.floor(round(south / CELL_DEGREES, 6))
    top = math.ceil(round(north / CELL_DEGREES, 6))
    return left * CELL_DEGREES, top * CELL_DEGREES, right - left, top - bottom


def _date_from_name(path: Path) -> date:
    match = _NAME.fullmatch(path.name)
    if match is None:
        raise ChirpsError(f"Unexpected CHIRPS source filename: {path.name}")
    try:
        return date(*(int(value) for value in match.groups()))
    except ValueError as exc:
        raise ChirpsError(f"Invalid CHIRPS source date: {path.name}") from exc


class ChirpsConnector:
    """Fetch daily GeoTIFFs, crop to the area and publish Parquet plus daily COGs."""

    dataset = DATASET
    version = VERSION

    def __init__(
        self,
        training_area: Bbox,
        start: date,
        end: date,
        download_dir: Path,
        window_reader: WindowReader = _read_remote_window,
    ) -> None:
        _grid(training_area)
        if end < start or start < date(1981, 1, 1):
            raise ChirpsError("Invalid CHIRPS date range")
        self.training_area = training_area
        self.start, self.end = start, end
        self.download_dir, self.window_reader = download_dir, window_reader

    def fetch(self) -> list[Path]:
        """Download one official CHIRPS v3 final RNL GeoTIFF per requested day."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for day in pd.date_range(self.start, self.end, freq="D"):
            filename = f"chirps-v3.0.rnl.{day:%Y.%m.%d}.tif"
            url = f"{BASE_URL}/{day:%Y}/{filename}"
            data = self.window_reader(url, self.training_area)
            if not data:
                raise ChirpsError(f"CHIRPS returned an empty response for {day:%Y-%m-%d}")
            path = self.download_dir / filename
            path.write_bytes(data)
            files.append(path)
        return files

    def _read(self, path: Path) -> np.ndarray:
        west, north, width, height = _grid(self.training_area)
        try:
            with rasterio.open(path) as source:
                transform = source.transform
                if source.crs is None or source.crs.to_epsg() != 4326 or source.count != 1:
                    raise ChirpsError(f"{path.name}: expected one EPSG:4326 rainfall band")
                if (
                    not math.isclose(transform.a, CELL_DEGREES, abs_tol=1e-7)
                    or not math.isclose(-transform.e, CELL_DEGREES, abs_tol=1e-7)
                    or not math.isclose(transform.b, 0, abs_tol=1e-9)
                    or not math.isclose(transform.d, 0, abs_tol=1e-9)
                ):
                    raise ChirpsError(f"{path.name}: expected an unrotated 0.05 degree grid")
                col = round((west - transform.c) / CELL_DEGREES)
                row = round((transform.f - north) / CELL_DEGREES)
                if (
                    abs(transform.c + col * CELL_DEGREES - west) > 1e-5
                    or abs(transform.f - row * CELL_DEGREES - north) > 1e-5
                ):
                    raise ChirpsError(f"{path.name}: grid is not aligned to 0.05 degrees")
                if col < 0 or row < 0 or col + width > source.width or row + height > source.height:
                    raise ChirpsError(f"{path.name}: training area is outside source coverage")
                values = source.read(1, window=Window(col, row, width, height)).astype("float32")
                missing = values == SOURCE_NODATA
                if source.nodata is not None:
                    missing |= values == source.nodata
                values[missing] = np.nan
        except rasterio.errors.RasterioIOError as exc:
            raise ChirpsError(f"{path.name}: invalid GeoTIFF") from exc
        if np.isinf(values).any() or np.any(values[~np.isnan(values)] < 0):
            raise ChirpsError(f"{path.name}: negative rainfall or non-finite values")
        return cast(np.ndarray, values)

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        """Make one row per day and grid-cell centre, with no-data as null."""
        expected = {day.date() for day in pd.date_range(self.start, self.end, freq="D")}
        actual = [_date_from_name(path) for path in raw_files]
        if len(actual) != len(set(actual)):
            raise ChirpsError("CHIRPS source has duplicate dates")
        if set(actual) != expected:
            raise ChirpsError("CHIRPS source has missing days or unexpected dates")
        west, north, width, height = _grid(self.training_area)
        longitudes = west + (np.arange(width) + 0.5) * CELL_DEGREES
        latitudes = north - (np.arange(height) + 0.5) * CELL_DEGREES
        rows = []
        for path in sorted(raw_files):
            day = _date_from_name(path)
            values = self._read(path)
            for row, latitude in enumerate(latitudes):
                for col, longitude in enumerate(longitudes):
                    rows.append(
                        (
                            day,
                            round(float(latitude), 6),
                            round(float(longitude), 6),
                            float(values[row, col]),
                        )
                    )
        return pd.DataFrame(rows, columns=["date", "latitude", "longitude", "rainfall_mm"])

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        """Write a table, a cloud-optimized GeoTIFF per day, and a provenance manifest."""
        out_dir.mkdir(parents=True, exist_ok=True)
        table = out_dir / f"{DATASET}.parquet"
        frame.to_parquet(table, index=False, engine="pyarrow")
        west, north, width, height = _grid(self.training_area)
        transform = from_origin(west, north, CELL_DEGREES, CELL_DEGREES)
        layers = []
        for path in sorted(raw_files):
            day = _date_from_name(path)
            values = self._read(path)
            stored = np.where(np.isnan(values), OUTPUT_NODATA, values).astype("float32")
            target = out_dir / "layers" / f"chirps-v3.0.rnl.{day:%Y.%m.%d}.tif"
            target.parent.mkdir(parents=True, exist_ok=True)
            with MemoryFile() as memory:
                with memory.open(
                    driver="GTiff",
                    width=width,
                    height=height,
                    count=1,
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=OUTPUT_NODATA,
                ) as raster:
                    raster.write(stored, 1)
                raster_copy(
                    memory.name,
                    target,
                    driver="COG",
                    compress="DEFLATE",
                    blocksize=256,
                    overview_resampling="nearest",
                )
            layers.append(
                {
                    "file": f"layers/{target.name}",
                    "checksum_sha256": _sha256(target),
                    "source_checksum_sha256": _sha256(path),
                }
            )
        west_bound, south_bound, east_bound, north_bound = self.training_area
        manifest = {
            "dataset": DATASET,
            "file": table.name,
            "source": "Climate Hazards Center CHIRPS v3 daily final RNL, "
            "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/",
            "source_version": VERSION,
            "license": "CC BY 4.0",
            "attribution": "Climate Hazards Center CHIRPS3 Data Repository, "
            "https://doi.org/10.15780/G2JQ0P",
            "rows": len(frame),
            "date_range": {"start": self.start.isoformat(), "end": self.end.isoformat()},
            "time_step": "daily",
            "spatial_resolution": "0.05 degrees, EPSG:4326",
            "crs": "EPSG:4326",
            "unit": "mm per day",
            "missing_value": "null",
            "bbox": {
                "west": west_bound,
                "south": south_bound,
                "east": east_bound,
                "north": north_bound,
            },
            "grid": {
                "west": west,
                "north": north,
                "width": width,
                "height": height,
                "cell_degrees": CELL_DEGREES,
            },
            "limitations": LIMITATIONS,
            "layers": layers,
            "checksum_sha256": _sha256(table),
            "source_checksum_sha256": {path.name: _sha256(path) for path in raw_files},
            "processing_version": __version__,
        }
        manifest_path = out_dir / f"{DATASET}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path
