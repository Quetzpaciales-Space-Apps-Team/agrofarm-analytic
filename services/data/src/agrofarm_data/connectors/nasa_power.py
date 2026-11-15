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
import time
from collections.abc import Callable
from datetime import date
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from agrofarm_data import __version__

DATASET = "climate_nasa_power_daily"
SOURCE_VERSION = "POWER Daily API, AG community (version not recorded in the file)"
ATTRIBUTION = "NASA Langley Research Center (LaRC) POWER Project"
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
        "source_version": SOURCE_VERSION,
        "license": "NASA open data. Acknowledge the NASA Langley Research Center POWER Project.",
        "attribution": ATTRIBUTION,
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


BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily"
Transport = Callable[[str], bytes]


def _download(url: str) -> bytes:
    """Fetch a POWER response with a timeout and bounded retries for transient failures."""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
                return bytes(response.read())
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise PowerError(f"POWER request failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 2:
                raise PowerError("POWER request failed after three attempts") from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _dates(start: date, end: date) -> pd.DatetimeIndex:
    if end < start:
        raise PowerError("End date must be on or after start date")
    return pd.date_range(start, end, freq="D")


def _payload(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PowerError("Invalid POWER JSON response") from exc
    if not isinstance(value, dict):
        raise PowerError("Invalid POWER JSON response")
    return value


def _series(raw: object, expected: pd.DatetimeIndex, missing: float) -> dict[pd.Timestamp, float]:
    if not isinstance(raw, dict) or not raw:
        raise PowerError("POWER response has an empty daily series or missing days")
    result: dict[pd.Timestamp, float] = {}
    for key, value in raw.items():
        try:
            day = pd.to_datetime(key, format="%Y%m%d", exact=True)
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PowerError("Invalid POWER date or value") from exc
        result[day] = float("nan") if number == missing else number
    if set(result) != set(expected):
        raise PowerError("POWER response has missing days or unexpected dates")
    return result


def _header(payload: dict[str, object]) -> tuple[float, str]:
    header = payload.get("header")
    if not isinstance(header, dict):
        raise PowerError("POWER response has no header")
    api = header.get("api")
    version = api.get("version") if isinstance(api, dict) else None
    return float(header.get("fill_value", -999)), f"POWER Daily API {version or 'unknown'}"


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise PowerError("Coordinates are outside valid latitude or longitude bounds")


class PowerPointConnector:
    """Download one AG point response and normalize its seven daily climate variables."""

    dataset = DATASET
    version = "daily-api-v2"

    def __init__(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        download_dir: Path,
        transport: Transport = _download,
    ) -> None:
        _validate_coordinates(latitude, longitude)
        _dates(start, end)
        self.latitude, self.longitude = latitude, longitude
        self.start, self.end = start, end
        self.download_dir, self.transport = download_dir, transport

    def fetch(self) -> list[Path]:
        query = urlencode(
            {
                "parameters": ",".join(RENAME),
                "community": "AG",
                "longitude": self.longitude,
                "latitude": self.latitude,
                "start": self.start.strftime("%Y%m%d"),
                "end": self.end.strftime("%Y%m%d"),
                "format": "JSON",
            }
        )
        self.download_dir.mkdir(parents=True, exist_ok=True)
        name = f"point_{self.latitude}_{self.longitude}_{self.start:%Y%m%d}_{self.end:%Y%m%d}.json"
        path = self.download_dir / name
        path.write_bytes(self.transport(f"{BASE_URL}/point?{query}"))
        return [path]

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        if len(raw_files) != 1:
            raise PowerError("Point response needs one raw file")
        payload = _payload(raw_files[0])
        missing, _ = _header(payload)
        geometry = payload.get("geometry")
        properties = payload.get("properties")
        if not isinstance(geometry, dict) or not isinstance(properties, dict):
            raise PowerError("Point response has no geometry or properties")
        coordinates = geometry.get("coordinates")
        parameters = properties.get("parameter")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) < 2
            or not isinstance(parameters, dict)
        ):
            raise PowerError("Point response has an empty parameter set or invalid coordinates")
        expected = _dates(self.start, self.end)
        frame = pd.DataFrame(
            {
                "date": expected,
                "latitude": float(coordinates[1]),
                "longitude": float(coordinates[0]),
            }
        )
        for source, clean in RENAME.items():
            values = _series(parameters.get(source), expected, missing)
            frame[clean] = [values[day] for day in expected]
        frame["wind_speed_kmh"] = (frame["wind_speed_kmh"] * MS_TO_KMH).round(3)
        _validate(frame)
        return frame[COLUMNS]

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        path = write_clean(frame, out_dir, raw_files[0])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _, manifest["source_version"] = _header(_payload(raw_files[0]))
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path


class PowerRegionalConnector:
    """Download one response per parameter and retain each parameter's native grid."""

    dataset = "climate_nasa_power_daily_regional"
    version = "daily-api-v2"

    def __init__(
        self,
        latitude_min: float,
        latitude_max: float,
        longitude_min: float,
        longitude_max: float,
        start: date,
        end: date,
        download_dir: Path,
        transport: Transport = _download,
    ) -> None:
        _validate_coordinates(latitude_min, longitude_min)
        _validate_coordinates(latitude_max, longitude_max)
        if latitude_max - latitude_min < 2 or longitude_max - longitude_min < 2:
            raise PowerError("POWER regional bounding box must span at least 2 degrees per axis")
        _dates(start, end)
        self.bounds = (latitude_min, latitude_max, longitude_min, longitude_max)
        self.start, self.end = start, end
        self.download_dir, self.transport = download_dir, transport

    def fetch(self) -> list[Path]:
        latitude_min, latitude_max, longitude_min, longitude_max = self.bounds
        self.download_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for parameter in RENAME:
            query = urlencode(
                {
                    "parameters": parameter,
                    "community": "AG",
                    "latitude-min": latitude_min,
                    "latitude-max": latitude_max,
                    "longitude-min": longitude_min,
                    "longitude-max": longitude_max,
                    "start": self.start.strftime("%Y%m%d"),
                    "end": self.end.strftime("%Y%m%d"),
                    "format": "JSON",
                }
            )
            name = (
                f"regional_{parameter}_{latitude_min}_{latitude_max}_"
                f"{longitude_min}_{longitude_max}_{self.start:%Y%m%d}_{self.end:%Y%m%d}.json"
            )
            path = self.download_dir / name
            path.write_bytes(self.transport(f"{BASE_URL}/regional?{query}"))
            files.append(path)
        return files

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        if len(raw_files) != len(RENAME):
            raise PowerError("Regional response needs one raw file per parameter")
        expected = _dates(self.start, self.end)
        rows = []
        for source, path in zip(RENAME, raw_files, strict=True):
            payload = _payload(path)
            missing, _ = _header(payload)
            features = payload.get("features")
            if not isinstance(features, list) or not features:
                raise PowerError("Regional response has empty features")
            for feature in features:
                try:
                    coordinates = feature["geometry"]["coordinates"]
                    raw = feature["properties"]["parameter"][source]
                    longitude, latitude = float(coordinates[0]), float(coordinates[1])
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise PowerError(
                        "Regional feature is missing coordinates or parameter"
                    ) from exc
                _validate_coordinates(latitude, longitude)
                for day, value in _series(raw, expected, missing).items():
                    normalized = round(value * MS_TO_KMH, 3) if source == "WS2M" else value
                    rows.append((day, latitude, longitude, RENAME[source], normalized))
        frame = pd.DataFrame(rows, columns=["date", "latitude", "longitude", "variable", "value"])
        if frame.duplicated(["date", "latitude", "longitude", "variable"]).any():
            raise PowerError("Regional response contains duplicate dates or cells")
        for variable, (low, high) in VALID_RANGE.items():
            bad = frame.loc[
                (frame["variable"] == variable) & ((frame["value"] < low) | (frame["value"] > high))
            ]
            if not bad.empty:
                raise PowerError(f"{variable} has values outside [{low}, {high}]")
        return frame.sort_values(["variable", "date", "latitude", "longitude"], ignore_index=True)

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        table = out_dir / f"{self.dataset}.parquet"
        frame.to_parquet(table, index=False, engine="pyarrow")
        _, source_version = _header(_payload(raw_files[0]))
        manifest = {
            "dataset": self.dataset,
            "file": table.name,
            "source": "NASA POWER Daily API, community AG, https://power.larc.nasa.gov",
            "source_version": source_version,
            "license": (
                "NASA open data. Acknowledge the NASA Langley Research Center POWER Project."
            ),
            "attribution": ATTRIBUTION,
            "rows": len(frame),
            "date_range": {"start": self.start.isoformat(), "end": self.end.isoformat()},
            "time_step": "daily",
            "bounds": dict(
                zip(
                    ("latitude_min", "latitude_max", "longitude_min", "longitude_max"),
                    self.bounds,
                    strict=True,
                )
            ),
            "spatial_resolution": "Native POWER source grids; solar radiation has a coarser grid",
            "crs": "EPSG:4326",
            "units": UNITS,
            "conversions": {"wind_speed_kmh": "WS2M in m/s multiplied by 3.6"},
            "missing_value": "null",
            "checksum_sha256": _sha256(table),
            "source_checksum_sha256": {path.name: _sha256(path) for path in raw_files},
            "processing_version": __version__,
        }
        path = out_dir / f"{self.dataset}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path
