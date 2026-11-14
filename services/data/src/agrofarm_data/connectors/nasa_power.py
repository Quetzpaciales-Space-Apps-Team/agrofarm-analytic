"""NASA POWER daily point series (MERRA-2 and CERES SYN1deg).

The source is the CSV of the POWER daily API for one point: a header block between
`-BEGIN HEADER-` and `-END HEADER-`, then one row per day with YEAR and DOY (day of year).
The header names the missing marker (-999), which becomes null here. Wind comes in m/s and
is converted to km/h, the unit the team agreed on.

POWER is a reanalysis on a grid of about 0.5 x 0.625 degrees: the point series is the
average of a cell of roughly 50 km, not a station measurement.
"""

import hashlib
import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from agrofarm_data import __version__

DATASET = "climate_nasa_power_daily"
MS_TO_KMH = 3.6
_BEGIN, _END = "-BEGIN HEADER-", "-END HEADER-"
_LOCATION = re.compile(r"Location:\s*latitude\s+(-?\d+(?:\.\d+)?)\s+longitude\s+(-?\d+(?:\.\d+)?)")
_MISSING = re.compile(r"missing source data.*?:\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)

# source column -> clean column (English, unit in the name)
RENAME = {
    "T2M": "temperature_mean_c",
    "T2M_MIN": "temperature_min_c",
    "T2M_MAX": "temperature_max_c",
    "PRECTOTCORR": "rainfall_mm",
    "RH2M": "relative_humidity_pct",
    "ALLSKY_SFC_SW_DWN": "solar_radiation_mj_m2",
    "WS2M": "wind_speed_kmh",
}
COLUMNS = ["date", "latitude", "longitude", *RENAME.values()]
UNITS = {
    "temperature_mean_c": "degrees C",
    "temperature_min_c": "degrees C",
    "temperature_max_c": "degrees C",
    "rainfall_mm": "mm per day",
    "relative_humidity_pct": "%",
    "solar_radiation_mj_m2": "MJ/m2 per day",
    "wind_speed_kmh": "km/h",
}
# Physical limits. A value outside them is a data error, not weather.
VALID_RANGE = {
    "temperature_mean_c": (-60.0, 60.0),
    "temperature_min_c": (-60.0, 60.0),
    "temperature_max_c": (-60.0, 60.0),
    "rainfall_mm": (0.0, 1000.0),
    "relative_humidity_pct": (0.0, 100.0),
    "solar_radiation_mj_m2": (0.0, 45.0),
    "wind_speed_kmh": (0.0, 300.0),
}


class PowerError(ValueError):
    """The file does not look like the NASA POWER daily CSV we know how to read."""


def read_power(path: Path) -> pd.DataFrame:
    """Read the point series, null the missing marker, convert wind and validate."""
    text = path.read_text(encoding="utf-8")
    if _BEGIN not in text or _END not in text:
        raise PowerError("The file has no POWER header block (-BEGIN HEADER- ... -END HEADER-).")
    header, body = text.split(_END, 1)
    location = _LOCATION.search(header)
    if location is None:
        raise PowerError("The header has no 'Location: latitude ... longitude ...' line.")
    marker = _MISSING.search(header)
    missing = float(marker.group(1)) if marker else -999.0

    raw = pd.read_csv(StringIO(body.lstrip()))
    absent = [column for column in ("YEAR", "DOY", *RENAME) if column not in raw.columns]
    if absent:
        raise PowerError(f"Missing columns: {', '.join(absent)}")
    if raw.empty:
        raise PowerError("The file has no data rows.")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                raw["YEAR"].astype(str) + raw["DOY"].astype(str).str.zfill(3), format="%Y%j"
            ),
            "latitude": float(location.group(1)),
            "longitude": float(location.group(2)),
        }
    )
    for source, clean in RENAME.items():
        values = pd.to_numeric(raw[source], errors="raise").astype("float64")
        frame[clean] = values.mask(values == missing)
    frame["wind_speed_kmh"] = (frame["wind_speed_kmh"] * MS_TO_KMH).round(3)
    frame = frame.sort_values("date", ignore_index=True)[COLUMNS]
    _validate(frame)
    return frame


def _validate(frame: pd.DataFrame) -> None:
    duplicated = frame.loc[frame["date"].duplicated(), "date"]
    if not duplicated.empty:
        first = duplicated.iloc[0].date().isoformat()
        raise PowerError(f"{len(duplicated)} duplicate dates, first: {first}")
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    gaps = expected.difference(pd.DatetimeIndex(frame["date"]))
    if len(gaps):
        raise PowerError(f"{len(gaps)} missing days, first: {gaps[0].date().isoformat()}")
    for column, (low, high) in VALID_RANGE.items():
        bad = frame.loc[(frame[column] < low) | (frame[column] > high), ["date", column]]
        if not bad.empty:
            day = bad.iloc[0]["date"].date().isoformat()
            raise PowerError(
                f"{column}: {len(bad)} values outside [{low}, {high}], first on {day}: "
                f"{bad.iloc[0][column]}"
            )
    unordered = frame.loc[
        (frame["temperature_min_c"] > frame["temperature_mean_c"])
        | (frame["temperature_mean_c"] > frame["temperature_max_c"]),
        "date",
    ]
    if not unordered.empty:
        day = unordered.iloc[0].date().isoformat()
        raise PowerError(
            f"{len(unordered)} days where minimum <= mean <= maximum temperature "
            f"does not hold, first: {day}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clean(frame: pd.DataFrame, out_dir: Path, source_file: Path) -> Path:
    """Write the clean table as Parquet plus its manifest. Returns the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{DATASET}.parquet"
    frame.to_parquet(table_path, index=False, engine="pyarrow")
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "NASA POWER Daily API, community AG (MERRA-2 meteorology, "
        "CERES SYN1deg solar radiation), https://power.larc.nasa.gov",
        "license": "NASA open data. Acknowledge the NASA Langley Research Center POWER Project.",
        "preliminary": False,
        "rows": int(len(frame)),
        "date_range": {
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
        },
        "time_step": "daily",
        "location": {
            "latitude": float(frame["latitude"].iloc[0]),
            "longitude": float(frame["longitude"].iloc[0]),
        },
        "spatial_resolution": "point taken from a grid cell of about 0.5 x 0.625 degrees "
        "(1 degree for solar radiation). Not a station measurement.",
        "crs": "EPSG:4326",
        "units": UNITS,
        "conversions": {"wind_speed_kmh": "WS2M in m/s multiplied by 3.6"},
        "missing_value": "null",
        "missing_fraction": {
            column: round(float(frame[column].isna().mean()), 4) for column in UNITS
        },
        "checksum_sha256": _sha256(table_path),
        "source_checksum_sha256": _sha256(source_file),
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
