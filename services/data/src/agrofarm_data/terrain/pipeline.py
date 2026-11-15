"""Terrain dataset: elevation, slope, aspect, flow accumulation and the two provisional risk
zones for a box, from Copernicus GLO-30 tiles. Writes a Parquet summary, one tiled GeoTIFF per
layer and a manifest.

Flow accumulation fills depressions first (priority flood with epsilon), then routes water to
the steepest of the eight neighbours (D8). See `analysis.flow_accumulation`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine

from agrofarm_data import __version__
from agrofarm_data.terrain import analysis, zones
from agrofarm_data.terrain.dem import MARGIN_DEG, Bbox, Dem, read_dem

DATASET = "terrain_copernicus_glo30"
LAYER_UNITS = {
    "elevation": "m above the EGM2008 geoid (surface model)",
    "slope": "degrees",
    "aspect": "degrees clockwise from north, direction the slope faces; null on flat cells",
    "flow_accumulation": "cells draining through each cell, itself included (1 cell is about 30 m)",
    "erosion_risk": "zone code: 0 low, 1 moderate, 2 high, 255 no data (provisional)",
    "waterlogging_risk": "zone code: 0 low, 1 moderate, 2 high, 255 no data (provisional)",
}
ZONE_LABELS = {zones.LOW: "low", zones.MODERATE: "moderate", zones.HIGH: "high"}
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
LICENSE = (
    "Copernicus DEM GLO-30 is free of charge for any use with attribution. Suggested notice: "
    "'Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and "
    "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
    "reserved.' This wording is from memory of the ESA terms: confirm it before publication."
)
LIMITATIONS = [
    "The 30 m DEM is a surface model: it includes buildings, trees and crops, not bare earth, "
    "so slope is inflated on canopy edges and a flow path can follow a tree line.",
    "Erosion and waterlogging zones are provisional heuristics from slope and flow "
    "accumulation only. They ignore soil, cover, rainfall and drainage works and are not "
    "agronomic truth. An agronomist must review the thresholds.",
    "Depressions are filled before routing (priority flood with epsilon), so pits and flats "
    "drain somewhere; real closed basins, ponds and wetlands are therefore not detected, and "
    "flow across filled flats is arbitrary within the flat.",
    "D8 sends all water to one neighbour: channels are one cell wide and flow on smooth "
    "slopes is over-concentrated.",
    "Flow accumulation counts only the cells read (the box plus a margin): catchments reaching "
    "beyond that are truncated, mostly near the margin edge.",
    "Cell size is converted to metres by latitude with a spherical approximation; vertical "
    "accuracy of GLO-30 is metres, which matters on gentle slopes.",
    "Accumulation limits are percentiles of the box, so zone shares change when the box changes.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_layers(dem: Dem) -> tuple[dict[str, NDArray[Any]], dict[str, float], Affine]:
    """All layers cropped to the box, the derived accumulation limits and the crop transform."""
    slope, aspect = analysis.slope_aspect(dem.values, dem.transform)
    accumulation = analysis.flow_accumulation(dem.values, dem.transform)
    rows = slice(int(dem.crop.row_off), int(dem.crop.row_off + dem.crop.height))
    cols = slice(int(dem.crop.col_off), int(dem.crop.col_off + dem.crop.width))
    cropped: dict[str, NDArray[Any]] = {
        "elevation": dem.values[rows, cols],
        "slope": slope[rows, cols],
        "aspect": aspect[rows, cols],
        "flow_accumulation": accumulation[rows, cols],
    }
    erosion, waterlogging, limits = zones.classify(cropped["slope"], cropped["flow_accumulation"])
    cropped["erosion_risk"] = erosion
    cropped["waterlogging_risk"] = waterlogging
    transform = dem.transform @ Affine.translation(cols.start, rows.start)
    return cropped, limits, transform


def _stats(values: NDArray[np.float64]) -> dict[str, float | None]:
    valid = values[~np.isnan(values)]
    if not valid.size:
        return dict.fromkeys(("mean", "p10", "p50", "p90", "min", "max"))
    p10, p50, p90 = (float(x) for x in np.percentile(valid, [10, 50, 90]))
    return {"mean": float(valid.mean()), "p10": p10, "p50": p50, "p90": p90,
            "min": float(valid.min()), "max": float(valid.max())}  # fmt: skip


def summarize(layers: dict[str, NDArray[Any]], transform: Affine) -> pd.DataFrame:
    """One row per continuous layer, per zone class and per aspect sector, over the box."""
    shape = layers["slope"].shape
    width_m, height_m = analysis.cell_sizes_m(transform, shape[0])
    area = np.broadcast_to((width_m * height_m)[:, None], shape) / 1e6  # km2 per cell
    rows: list[dict[str, Any]] = []

    def row(
        layer: str, kind: str, label: str | None, code: int | None, mask: Any, valid: Any
    ) -> dict[str, Any]:
        # mask: cells of this class; valid: cells that have data
        total = float(area[valid].sum())
        return {
            "layer": layer, "kind": kind, "class_code": code, "class_label": label,
            "unit": LAYER_UNITS[layer], "cells": int(mask.sum()),
            "valid_fraction": round(float(valid.sum() / valid.size), 4),
            "area_km2": float(area[mask].sum()),
            "area_share": float(area[mask].sum() / total) if total else None,
        }  # fmt: skip

    empty = dict.fromkeys(("mean", "p10", "p50", "p90", "min", "max"))
    for name in ("elevation", "slope", "flow_accumulation"):
        values = layers[name].astype("float64")
        valid = ~np.isnan(values)
        entry = row(name, "continuous", None, None, valid, valid)
        entry["area_share"] = None
        rows.append({**entry, **_stats(values)})
    aspect = layers["aspect"]
    has_slope = ~np.isnan(layers["slope"])
    with np.errstate(invalid="ignore"):
        sector_of = np.floor(((aspect + 22.5) % 360.0) / 45.0)  # 0 is N, 1 is NE, ...
    for index, sector in enumerate(SECTORS):
        mask = (sector_of == index) & has_slope
        rows.append({**row("aspect", "category", sector, index, mask, has_slope), **empty})
    flat = np.isnan(aspect) & has_slope
    rows.append({**row("aspect", "category", "flat", None, flat, has_slope), **empty})
    for name in ("erosion_risk", "waterlogging_risk"):
        zone = layers[name]
        valid = zone != zones.NODATA
        for code, label in ZONE_LABELS.items():
            rows.append({**row(name, "zone", label, code, zone == code, valid), **empty})
    return pd.DataFrame(rows)


def _write_layer(path: Path, values: NDArray[Any], transform: Affine, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = "uint8" if values.dtype == np.uint8 else "float32"
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1], "count": 1,
        "dtype": dtype, "crs": "EPSG:4326", "transform": transform, "nodata": nodata,
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as out:
        out.write(values.astype(dtype), 1)


def write_clean(raw_dir: Path, out_dir: Path, bbox: Bbox, area_name: str) -> Path:
    """Write the table, one tiled GeoTIFF per layer and the manifest. Returns the manifest."""
    dem = read_dem(raw_dir, bbox)
    layers, limits, transform = compute_layers(dem)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = summarize(layers, transform)
    table_path = out_dir / f"{DATASET}.parquet"
    table.to_parquet(table_path, index=False, engine="pyarrow")
    layer_entries = []
    for name in LAYER_UNITS:
        target = out_dir / "layers" / f"{name}.tif"
        is_zone = layers[name].dtype == np.uint8
        _write_layer(target, layers[name], transform, zones.NODATA if is_zone else float("nan"))
        layer_entries.append({"file": f"layers/{name}.tif", "checksum_sha256": _sha256(target)})
    tile_entries = []
    for path in dem.tiles:
        with rasterio.open(path) as source:
            bounds = list(source.bounds)
        tile_entries.append({"file": path.name, "sha256": _sha256(path), "bounds": bounds})
    west, south, east, north = bbox
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "Copernicus DEM GLO-30 (ESA, Airbus), 1 arc second surface model",
        "license": LICENSE,
        "license_verified": False,
        "license_note": "The tiles carry no license or attribution metadata; the notice above "
        "was not checked against the ESA terms.",
        "area_name": area_name,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "rows": int(len(table)),
        "time_step": "static",
        "spatial_resolution": "1 arc second (about 30 m, 0.000278 degrees)",
        "crs": "EPSG:4326",
        "margin_deg": MARGIN_DEG,
        "units": LAYER_UNITS,
        "missing_value": "null (NaN in float layers, 255 in zone layers)",
        "depressions": "filled (priority flood with epsilon) before D8 flow routing",
        "flow_routing": "D8, steepest descent, distances in metres",
        "thresholds": {
            "status": "Provisional heuristics, team assumptions, pending agronomist review",
            "provenance": zones.PROVENANCE,
            "erosion_slope_moderate_deg": zones.EROSION_SLOPE_MODERATE_DEG,
            "erosion_slope_high_deg": zones.EROSION_SLOPE_HIGH_DEG,
            "convergence_percentile": zones.CONVERGENCE_PERCENTILE,
            "min_convergence_cells": zones.MIN_CONVERGENCE_CELLS,
            "waterlogging_max_slope_deg": zones.WATERLOGGING_MAX_SLOPE_DEG,
            "waterlogging_moderate_percentile": zones.WATERLOGGING_MODERATE_PERCENTILE,
            "waterlogging_high_percentile": zones.WATERLOGGING_HIGH_PERCENTILE,
            "derived_for_this_box": limits,
        },
        "limitations": LIMITATIONS,
        "tiles": tile_entries,
        "layers": layer_entries,
        "checksum_sha256": _sha256(table_path),
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
