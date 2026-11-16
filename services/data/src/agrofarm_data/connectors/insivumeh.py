"""INSIVUMEH daily climate (station INSIVUMEH, Guatemala City).

The source is an Excel workbook with one data sheet per period (named like `1995-2022`),
three metadata rows, a blank row and the header on row 5. Missing values are written as
-99.9 and become null here. The data is marked preliminary by the provider.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from agrofarm_data import __version__

DATASET = "climate_insivumeh_daily"
SENTINEL = -99.9
SOURCE_VERSION = "Team workbook, sheets 1995-2022 and 2023-2025 (no release number)"
ATTRIBUTION = "INSIVUMEH, Guatemala"
HEADER_ROW = 5
_DATA_SHEET = re.compile(r"^\d{4}\s*[-–]\s*\d{4}$")

# source column -> clean column (English, unit in the name)
RENAME = {
    "FECHA": "date",
    "NOMBRE_ESTACIÓN": "station",
    "TEMPERATURA_MEDIA": "temperature_mean_c",
    "PRECIPITACIÓN": "rainfall_mm",
    "BRILLO_SOLAR": "sunshine_hours",
    "NUBOSIDAD": "cloud_cover_oktas",
    "VELOCIDAD_VIENTO": "wind_speed_kmh",
    "HUMEDAD_RELATIVA": "relative_humidity_pct",
}
COLUMNS = list(RENAME.values())
UNITS = {
    "temperature_mean_c": "degrees C",
    "rainfall_mm": "mm per day",
    "sunshine_hours": "hours of sunshine per day",
    "cloud_cover_oktas": "oktas",
    "wind_speed_kmh": "km/h",
    "relative_humidity_pct": "%",
}
# Physical limits. A value outside them is a data error, not weather.
VALID_RANGE = {
    "temperature_mean_c": (-10.0, 50.0),
    "rainfall_mm": (0.0, 1000.0),
    "sunshine_hours": (0.0, 24.0),
    "cloud_cover_oktas": (0.0, 8.0),
    "wind_speed_kmh": (0.0, 300.0),
    "relative_humidity_pct": (0.0, 100.0),
}


class InsivumehError(ValueError):
    """The workbook does not look like the INSIVUMEH file we know how to read."""


def read_insivumeh(path: Path) -> pd.DataFrame:
    """Read every data sheet, merge them, null the sentinel and validate."""
    parts: list[pd.DataFrame] = []
    with pd.ExcelFile(path, engine="openpyxl") as book:
        names = [str(name) for name in book.sheet_names if _DATA_SHEET.match(str(name).strip())]
        sheets = pd.read_excel(book, sheet_name=names, header=HEADER_ROW - 1) if names else {}
    for name, sheet in sheets.items():
        if list(sheet.columns) != list(RENAME):
            raise InsivumehError(
                f"Sheet {name!r}: unexpected header on row {HEADER_ROW}: {list(sheet.columns)}"
            )
        parts.append(sheet.rename(columns=RENAME))
    if not parts:
        raise InsivumehError("The workbook has no data sheets (expected names like '1995-2022').")

    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["station"] = frame["station"].astype("string")
    for column in UNITS:
        values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        frame[column] = values.mask(values == SENTINEL)
    frame = frame.sort_values("date", ignore_index=True)[COLUMNS]
    _validate(frame)
    return frame


def _validate(frame: pd.DataFrame) -> None:
    duplicated = frame.loc[frame["date"].duplicated(), "date"]
    if not duplicated.empty:
        first = duplicated.iloc[0].date().isoformat()
        raise InsivumehError(f"{len(duplicated)} duplicate dates, first: {first}")
    for column, (low, high) in VALID_RANGE.items():
        bad = frame.loc[(frame[column] < low) | (frame[column] > high), ["date", column]]
        if not bad.empty:
            day = bad.iloc[0]["date"].date().isoformat()
            raise InsivumehError(
                f"{column}: {len(bad)} values outside [{low}, {high}], first on {day}: "
                f"{bad.iloc[0][column]}"
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
        "source": "INSIVUMEH, station INSIVUMEH, Guatemala City (team workbook)",
        "source_version": SOURCE_VERSION,
        "license": "Not stated by the provider. Attribution: INSIVUMEH, Guatemala.",
        "attribution": ATTRIBUTION,
        "preliminary": True,
        "rows": int(len(frame)),
        "date_range": {
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
        },
        "time_step": "daily",
        "spatial_resolution": "single station (point)",
        "crs": "EPSG:4326",
        "units": UNITS,
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
