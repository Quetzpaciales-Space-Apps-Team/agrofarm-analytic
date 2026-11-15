"""ISRIC SoilGrids 2.0 static layers (mean prediction, 250 m, six standard depths).

Each source file is an int16 GeoTIFF named `<property>_<depth>_mean.tif`. SoilGrids stores
conventional units (for example pH x 10, clay in g/kg), which are scaled here to the units
agronomists use. The files in the team download carry no nodata flag: a cell value of 0 means
"no data" (built-up and water areas are masked), so 0 becomes null, never a measurement.

SoilGrids is a machine learning prediction, not a laboratory analysis.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine
from rasterio.windows import from_bounds

from agrofarm_data import __version__

DATASET = "soil_soilgrids_static"
SOURCE_VERSION = "2.0"
ATTRIBUTION = "ISRIC - World Soil Information, SoilGrids 2.0"
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
Bbox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)

# property -> (divisor from stored value to clean unit, clean unit, valid range in clean unit)
PROPERTIES: dict[str, tuple[float, str, tuple[float, float]]] = {
    "phh2o": (10.0, "pH", (2.0, 11.0)),
    "soc": (10.0, "g/kg", (0.0, 1000.0)),
    "clay": (10.0, "%", (0.0, 100.0)),
    "sand": (10.0, "%", (0.0, 100.0)),
    "silt": (10.0, "%", (0.0, 100.0)),
    "nitrogen": (100.0, "g/kg", (0.0, 100.0)),
    "cec": (10.0, "cmol(c)/kg", (0.0, 200.0)),
    "bdod": (100.0, "kg/dm3", (0.1, 2.5)),
}
UNITS = {name: unit for name, (_, unit, _) in PROPERTIES.items()}
LIMITATIONS = [
    "SoilGrids is a machine learning prediction at 250 m, not a laboratory analysis.",
    "A soil analysis of the farm overrides these values.",
    "The source stores 0 for no data (built-up areas and water are masked). 0 is read as "
    "no data and becomes null. Check valid_fraction before using a value.",
]


class SoilGridsError(ValueError):
    """The SoilGrids files are missing, do not cover the area or hold impossible values."""


def _path(raw_dir: Path, prop: str, depth: str) -> Path:
    return raw_dir / f"{prop}_{depth}_mean.tif"


def read_layer(path: Path, prop: str, bbox: Bbox) -> tuple[NDArray[np.float32], Affine]:
    """Read one layer cropped to the box, scaled to clean units, no-data as NaN."""
    if not path.is_file():
        raise SoilGridsError(f"Missing layer: {path.name}")
    divisor, _, (low, high) = PROPERTIES[prop]
    west, south, east, north = bbox
    with rasterio.open(path) as source:
        left, bottom, right, top = source.bounds
        if east <= left or west >= right or north <= bottom or south >= top:
            raise SoilGridsError(f"The box {bbox} is outside the raster {path.name}")
        window = from_bounds(west, south, east, north, source.transform).round_offsets()
        window = window.round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, source.width, source.height))
        stored = source.read(1, window=window)
        transform = source.window_transform(window)
    values = np.where(stored == 0, np.nan, stored / divisor).astype("float32")
    valid = values[~np.isnan(values)]
    if valid.size and (valid.min() < low or valid.max() > high):
        raise SoilGridsError(
            f"{prop}: values outside [{low}, {high}] in {path.name}: "
            f"{float(valid.min())} to {float(valid.max())}"
        )
    return values, transform


def summarize(raw_dir: Path, bbox: Bbox) -> pd.DataFrame:
    """One row per property and depth with mean, spread and the share of cells with data."""
    rows = []
    for prop in PROPERTIES:
        for depth in DEPTHS:
            values, _ = read_layer(_path(raw_dir, prop, depth), prop, bbox)
            valid = values[~np.isnan(values)]
            rows.append(
                {
                    "property": prop,
                    "depth": depth,
                    "unit": UNITS[prop],
                    "mean": float(valid.mean()) if valid.size else None,
                    "p10": float(np.percentile(valid, 10)) if valid.size else None,
                    "p90": float(np.percentile(valid, 90)) if valid.size else None,
                    "cells": int(values.size),
                    "valid_fraction": round(float(valid.size / values.size), 4),
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_layer(path: Path, values: NDArray[np.float32], transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1], "count": 1,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform, "nodata": float("nan"),
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as out:
        out.write(values, 1)


def write_clean(raw_dir: Path, out_dir: Path, bbox: Bbox, area_name: str) -> Path:
    """Write the table, one tiled GeoTIFF per layer and the manifest. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = summarize(raw_dir, bbox)
    table_path = out_dir / f"{DATASET}.parquet"
    table.to_parquet(table_path, index=False, engine="pyarrow")
    layers = []
    source_checksums = {}
    for prop in PROPERTIES:
        for depth in DEPTHS:
            source = _path(raw_dir, prop, depth)
            values, transform = read_layer(source, prop, bbox)
            target = out_dir / "layers" / f"{prop}_{depth}.tif"
            _write_layer(target, values, transform)
            source_checksums[source.name] = _sha256(source)
            layers.append(
                {
                    "file": f"layers/{target.name}",
                    "source_checksum_sha256": _sha256(source),
                    "checksum_sha256": _sha256(target),
                }  # fmt: skip
            )
    west, south, east, north = bbox
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "ISRIC SoilGrids 2.0 (mean), https://soilgrids.org",
        "source_version": SOURCE_VERSION,
        "license": "CC BY 4.0. Attribute ISRIC - World Soil Information.",
        "attribution": ATTRIBUTION,
        "area_name": area_name,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "rows": int(len(table)),
        "time_step": "static",
        "spatial_resolution": "250 m prediction (about 0.00225 degrees)",
        "crs": "EPSG:4326",
        "depths": list(DEPTHS),
        "units": UNITS,
        "missing_value": "null (source value 0 is no data)",
        "limitations": LIMITATIONS,
        "layers": layers,
        "source_checksum_sha256": source_checksums,
        "checksum_sha256": _sha256(table_path),
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
