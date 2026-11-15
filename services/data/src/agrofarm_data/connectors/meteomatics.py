"""Meteomatics point forecast (daily values) with a disk cache and a daily request budget.

The hackathon account allows 1,000 queries per day and ends with the event, so every
response is cached and a counter refuses requests beyond a daily limit. One forecast is one
query, whatever the number of parameters and days.

Credentials come from the caller (the CLI reads METEOMATICS_USERNAME and METEOMATICS_PASSWORD).
They go only into the Authorization header: never into the URL, the cache, the manifest, a log
or an error message. The account name that the API echoes back is removed before caching.

A forecast is a model output for future days, not an observation.
"""

import base64
import datetime as dt
import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from agrofarm_data import __version__

DATASET = "forecast_meteomatics_point_daily"
API = "https://api.meteomatics.com"
DEFAULT_DAILY_LIMIT = 500  # half of the 1,000 queries per day of the account, as a margin
MAX_DAYS = 14
MISSING = -999
USER_AGENT = "agrofarm-data-connector/0.1"

# API parameter -> (clean column, factor from the API unit to the clean unit)
PARAMETERS: dict[str, tuple[str, float]] = {
    "t_mean_2m_24h:C": ("temperature_mean_c", 1.0),
    "t_min_2m_24h:C": ("temperature_min_c", 1.0),
    "t_max_2m_24h:C": ("temperature_max_c", 1.0),
    "precip_24h:mm": ("rainfall_mm", 1.0),
    "wind_speed_mean_10m_24h:kmh": ("wind_speed_kmh", 1.0),
    "relative_humidity_mean_2m_24h:p": ("relative_humidity_pct", 1.0),
    "global_rad_24h:J": ("solar_radiation_mj_m2", 1e-6),  # J/m2 to MJ/m2
}
COLUMNS = ["date", "latitude", "longitude", *(column for column, _ in PARAMETERS.values())]
UNITS = {
    "temperature_mean_c": "degrees C",
    "temperature_min_c": "degrees C",
    "temperature_max_c": "degrees C",
    "rainfall_mm": "mm per day",
    "wind_speed_kmh": "km/h (at 10 m)",
    "relative_humidity_pct": "%",
    "solar_radiation_mj_m2": "MJ/m2 per day",
}
LIMITATIONS = [
    "This is a forecast (model output for future days), not an observation. "
    "It gets less reliable with every day of lead time.",
    "Each value covers the 24 hours that end at 00:00 UTC of the date, not the local day.",
    "The account is a hackathon account: 1,000 queries per day, and it ends with the event. "
    "Nothing in the product may depend on it.",
]

Http = Callable[[str, str, str], str]


class MeteomaticsError(RuntimeError):
    """The API could not be used or answered something that is not a forecast."""


class BudgetExceededError(MeteomaticsError):
    """The daily request limit is spent. Cached answers stay available."""


def _http_get(url: str, username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = urllib.request.Request(  # noqa: S310
        url, headers={"Authorization": f"Basic {token}", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        body: str = response.read().decode("utf-8")
    return body


class MeteomaticsClient:
    def __init__(
        self,
        cache_dir: Path,
        username: str,
        password: str,
        *,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
        http: Http = _http_get,
        today: Callable[[], dt.date] = lambda: dt.datetime.now(dt.UTC).date(),
    ) -> None:
        self._cache_dir = cache_dir
        self._username = username
        self._password = password
        self._daily_limit = daily_limit
        self._http = http
        self._today = today
        self.last_response_file: Path | None = None

    # ---- budget -------------------------------------------------------------------------

    def _budget_file(self) -> Path:
        return self._cache_dir / "budget.json"

    def requests_today(self) -> int:
        path = self._budget_file()
        if not path.is_file():
            return 0
        counts: dict[str, int] = json.loads(path.read_text("utf-8"))
        return int(counts.get(self._today().isoformat(), 0))

    def _spend_one_request(self) -> None:
        used = self.requests_today()
        if used >= self._daily_limit:
            raise BudgetExceededError(
                f"Daily limit of {self._daily_limit} Meteomatics requests reached. "
                f"Cached answers stay available; new requests wait until tomorrow (UTC)."
            )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_file().write_text(
            json.dumps({self._today().isoformat(): used + 1}) + "\n", encoding="utf-8"
        )

    # ---- requests -----------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        for secret in (self._password, self._username):
            if secret:
                text = text.replace(secret, "<hidden>")
        return text

    def _get(self, url: str) -> dict[str, Any]:
        """Return the JSON answer for the URL, from the cache when it is there."""
        name = hashlib.sha256(url.encode()).hexdigest()
        path = self._cache_dir / "responses" / f"{name}.json"
        self.last_response_file = path
        if path.is_file():
            cached: dict[str, Any] = json.loads(path.read_text("utf-8"))
            return cached
        if not self._username or not self._password:
            raise MeteomaticsError(
                "Missing credentials: set METEOMATICS_USERNAME and METEOMATICS_PASSWORD."
            )
        self._spend_one_request()  # counted before the call: a failed call still used quota
        try:
            body = self._http(url, self._username, self._password)
        except OSError as error:
            raise MeteomaticsError(f"Request failed: {self._scrub(str(error))}") from None
        try:
            answer = json.loads(body)
        except json.JSONDecodeError:
            raise MeteomaticsError("The answer is not JSON.") from None
        if not isinstance(answer, dict) or answer.get("status") != "OK":
            message = answer.get("message", "unknown") if isinstance(answer, dict) else "unknown"
            raise MeteomaticsError(f"The API reported an error: {self._scrub(str(message))}")
        if not answer.get("data"):
            raise MeteomaticsError("The answer has no data.")
        answer.pop("user", None)  # the account name is not ours to store
        answer["request_url"] = url
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer, indent=1) + "\n", encoding="utf-8")
        return answer

    def forecast(self, lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """Daily point forecast for `days` days starting today (UTC). One query."""
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Not a valid point: {lat}, {lon}")
        if not 1 <= days <= MAX_DAYS:
            raise ValueError(f"days must be between 1 and {MAX_DAYS}")
        start = self._today()
        end = start + dt.timedelta(days=days - 1)
        url = (
            f"{API}/{start.isoformat()}T00:00:00Z--{end.isoformat()}T00:00:00Z:P1D/"
            f"{','.join(PARAMETERS)}/{lat:.4f},{lon:.4f}/json"
        )
        answer = self._get(url)
        series: dict[str, dict[str, float | None]] = {}
        for item in answer["data"]:
            if item["parameter"] not in PARAMETERS:
                continue
            column, factor = PARAMETERS[item["parameter"]]
            series[column] = {
                entry["date"][:10]: None if entry["value"] == MISSING else entry["value"] * factor
                for entry in item["coordinates"][0]["dates"]
            }
        absent = [column for column, _ in PARAMETERS.values() if column not in series]
        if absent:
            raise MeteomaticsError(f"The answer lacks: {', '.join(absent)}")
        frame = pd.DataFrame(series).astype("float64")
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index().rename_axis("date").reset_index()
        frame["latitude"], frame["longitude"] = float(lat), float(lon)
        return frame[COLUMNS]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clean(frame: pd.DataFrame, out_dir: Path, response_file: Path | None) -> Path:
    """Write the forecast table as Parquet plus its manifest. Returns the manifest path."""
    if response_file is None or not response_file.is_file():
        raise MeteomaticsError("No cached response to describe: call forecast() first.")
    response = json.loads(response_file.read_text("utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{DATASET}.parquet"
    frame.to_parquet(table_path, index=False, engine="pyarrow")
    manifest = {
        "dataset": DATASET,
        "file": table_path.name,
        "source": "Meteomatics Weather API (point forecast, model mix), https://www.meteomatics.com",
        "retrieved_at": str(response.get("dateGenerated", ""))[:10],
        "license": "Meteomatics terms for the NASA Space Apps hackathon account. "
        "Not verified for redistribution: keep the data out of public repositories.",
        "attribution": "Weather data provided by Meteomatics",
        "rows": int(len(frame)),
        "date_range": {
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
        },
        "time_step": "daily (24 hours ending at 00:00 UTC of the date)",
        "location": {
            "latitude": float(frame["latitude"].iloc[0]),
            "longitude": float(frame["longitude"].iloc[0]),
        },
        "spatial_resolution": "point, downscaled by the provider (not a station measurement)",
        "crs": "EPSG:4326",
        "units": UNITS,
        "missing_value": "null",
        "limitations": LIMITATIONS,
        "checksum_sha256": _sha256(table_path),
        "source_checksum_sha256": _sha256(response_file),
        "processing_version": __version__,
    }
    manifest_path = out_dir / f"{DATASET}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
