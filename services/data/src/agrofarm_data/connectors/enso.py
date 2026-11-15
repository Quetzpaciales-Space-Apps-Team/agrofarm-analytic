"""ENSO indices from NOAA CPC: classic ONI and relative RONI.

Both files have one row per overlapping three-month season (`DJF 1950`, `JFM 1950`, ...).
Each season is assigned to its centre month: DJF of year Y is centred on January Y, and
NDJ of year Y on December Y. Each month is marked El Nino (index at or above +0.5 degrees
C), La Nina (at or below -0.5) or neutral. The phase follows RONI, NOAA's official index;
the ONI-based phase is kept alongside for comparison.
"""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from agrofarm_data import __version__

DATASET = "enso_monthly"
THRESHOLD_C = 0.5
MISSING_BELOW = -90.0  # NOAA writes missing values as -99.9 style markers
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
SOURCE_VERSION = "NOAA CPC oni.ascii.txt and RONI.ascii.txt (no release number)"
ATTRIBUTION = "NOAA Climate Prediction Center"
COLUMNS = ["month", "oni", "roni", "phase", "phase_oni"]


class EnsoError(ValueError):
    """The file does not look like a NOAA CPC ONI or RONI text file."""


def classify_phase(value: float) -> str | None:
    if value is None or math.isnan(value):
        return None
    if value >= THRESHOLD_C:
        return "el_nino"
    if value <= -THRESHOLD_C:
        return "la_nina"
    return "neutral"


def _read_index(path: Path, name: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()[1:], start=2):
        parts = line.split()
        if not parts:
            continue
        season, year, anomaly = parts[0], parts[1], parts[-1]
        if season not in SEASONS:
            raise EnsoError(f"{path.name} line {number}: unknown season code {season!r}")
        value = float(anomaly)
        month = pd.Timestamp(year=int(year), month=SEASONS.index(season) + 1, day=1)
        values[month] = float("nan") if value < MISSING_BELOW else value
    if not values:
        raise EnsoError(f"{path.name} has no rows")
    return pd.Series(values, name=name, dtype="float64")


def read_enso(oni_path: Path, roni_path: Path) -> pd.DataFrame:
    frame = pd.concat([_read_index(oni_path, "oni"), _read_index(roni_path, "roni")], axis=1)
    frame = frame.sort_index().rename_axis("month").reset_index()
    frame["phase"] = [classify_phase(value) for value in frame["roni"]]
    frame["phase_oni"] = [classify_phase(value) for value in frame["oni"]]
    return frame[COLUMNS]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clean(frame: pd.DataFrame, out_dir: Path, oni_file: Path, roni_file: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{DATASET}.parquet"
    frame.to_parquet(table_path, index=False, engine="pyarrow")
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "NOAA Climate Prediction Center: oni.ascii.txt and RONI.ascii.txt",
        "source_version": SOURCE_VERSION,
        "license": "Public domain (US Government work). Attribution: NOAA CPC.",
        "attribution": ATTRIBUTION,
        "rows": int(len(frame)),
        "date_range": {
            "start": frame["month"].min().date().isoformat(),
            "end": frame["month"].max().date().isoformat(),
        },
        "time_step": "monthly (centre month of each three-month season)",
        "spatial_resolution": "one global value",
        "units": {"oni": "degrees C anomaly", "roni": "degrees C anomaly"},
        "threshold_c": THRESHOLD_C,
        "phase_index": "roni",
        "missing_value": "null",
        "checksum_sha256": _sha256(table_path),
        "source_checksum_sha256": {"oni": _sha256(oni_file), "roni": _sha256(roni_file)},
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
