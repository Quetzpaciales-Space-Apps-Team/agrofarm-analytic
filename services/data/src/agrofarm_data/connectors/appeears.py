"""AppEEARS connector: SMAP L4 soil moisture and MODIS and VIIRS vegetation indices for a polygon.

AppEEARS (NASA LP DAAC) samples products for an area and returns the result as a bundle of files.
A request takes minutes, so the job never waits for it: the first run submits the task and stores
its id, later runs poll it once, and a run that finds it `done` downloads the bundle. `TaskPending`
tells the caller to run the same command again. The state file holds the task id and a hash of the
request. It never holds a token or a password.

Authentication: AppEEARS does not accept the Earthdata profile token (it answers 403, even on its
public endpoints, when one is sent). It issues its own session token for `POST /login` with the
Earthdata user name and password, read from `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`. The
session token lives in memory for one run. Nothing secret is logged, stored or written
to a manifest.

Product and layer ids were confirmed against the catalog on 2026-11-15 (`confirm_layers` repeats the
check on every run and the manifest records what the catalog said). SMAP L4 is `SPL4SMGP.008`, not
the 007 that older notes named.
"""

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"
USER_AGENT = "agrofarm-data-connector/0.1"
ATTEMPTS = 3
PENDING_STATUSES = {"pending", "processing", "queued"}
FAILED_STATUSES = {"error", "expired", "deleted"}

Bbox = tuple[float, float, float, float]  # west, south, east, north, EPSG:4326


class AppEearsError(ValueError):
    """The request, the catalog or the service gave something that cannot be used."""


class TaskPending(RuntimeError):  # noqa: N818 -- a status to act on, not a failure
    """The task exists and is not finished. Run the same command again later."""

    def __init__(self, task_id: str, status: str) -> None:
        super().__init__(f"AppEEARS task {task_id} is {status}; run the same command again later")
        self.task_id = task_id
        self.status = status


@dataclass(frozen=True)
class Layer:
    name: str
    role: str  # what the layer means: ndvi, evi, quality, soil_moisture_surface, ...


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset: str
    product: str
    platform: str
    resolution: str
    time_step: str
    layers: tuple[Layer, ...]
    license_note: str


SPECS: dict[str, SourceSpec] = {
    "smap-l4": SourceSpec(
        key="smap-l4",
        dataset="appeears_smap_l4_soil_moisture",
        product="SPL4SMGP.008",
        platform="SMAP",
        resolution="9 km, EASE-Grid 2.0",
        time_step="daily (the product itself is 3-hourly)",
        layers=(
            Layer("Geophysical_Data_sm_surface", "soil_moisture_surface"),
            Layer("Geophysical_Data_sm_rootzone", "soil_moisture_rootzone"),
        ),
        license_note="NASA Earth science data. License terms of this product were not checked.",
    ),
    "modis-13q1": SourceSpec(
        key="modis-13q1",
        dataset="appeears_modis_13q1_vegetation",
        product="MOD13Q1.061",
        platform="Terra MODIS",
        resolution="250 m",
        time_step="16 days",
        layers=(
            Layer("_250m_16_days_NDVI", "ndvi"),
            Layer("_250m_16_days_EVI", "evi"),
            Layer("_250m_16_days_pixel_reliability", "quality"),
        ),
        license_note="NASA Earth science data. License terms of this product were not checked.",
    ),
    "viirs-vnp13a1": SourceSpec(
        key="viirs-vnp13a1",
        dataset="appeears_viirs_vnp13a1_vegetation",
        product="VNP13A1.002",
        platform="Suomi NPP VIIRS",
        resolution="500 m",
        time_step="16 days",
        layers=(
            Layer("500_m_16_days_NDVI", "ndvi"),
            Layer("500_m_16_days_EVI", "evi"),
            Layer("500_m_16_days_pixel_reliability", "quality"),
        ),
        license_note="NASA Earth science data. License terms of this product were not checked.",
    ),
}


class Transport(Protocol):
    """The one seam to the network. Tests replace it with recorded answers."""

    def json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        basic: tuple[str, str] | None = None,
    ) -> Any: ...

    def download(self, path: str, token: str, destination: Path) -> None: ...


class UrllibTransport:
    """Standard-library transport with a bounded retry on network errors and 5xx answers."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep = sleep

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        token: str | None,
        basic: tuple[str, str] | None,
    ) -> urllib.request.Request:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif method == "POST":
            data = b""
        if basic is not None:
            pair = base64.b64encode(f"{basic[0]}:{basic[1]}".encode()).decode("ascii")
            headers["Authorization"] = f"Basic {pair}"
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return urllib.request.Request(  # noqa: S310 -- fixed https base url
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )

    def _open(self, request: urllib.request.Request, method: str, path: str) -> Any:
        last = ""
        for attempt in range(ATTEMPTS):
            try:
                return urllib.request.urlopen(request, timeout=120)  # noqa: S310
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    # Only the status code is reported: the body of an error may echo input.
                    raise AppEearsError(f"HTTP {exc.code} on {method} {path}") from None
                last = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError) as exc:
                last = type(exc).__name__
            if attempt < ATTEMPTS - 1:
                self.sleep(2.0 * (attempt + 1))
        raise AppEearsError(f"{method} {path} failed after {ATTEMPTS} attempts ({last})")

    def json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        basic: tuple[str, str] | None = None,
    ) -> Any:
        with self._open(self._request(method, path, body, token, basic), method, path) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise AppEearsError(f"{method} {path} did not return JSON") from None

    def download(self, path: str, token: str, destination: Path) -> None:
        request = self._request("GET", path, None, token, None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with self._open(request, "GET", path) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        partial.rename(destination)


def credentials_from_env(env: Mapping[str, str]) -> tuple[str, str]:
    """The Earthdata user name and password. A missing one is named, never printed."""
    for key in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        if not env.get(key):
            raise AppEearsError(
                f"{key} is not set; AppEEARS needs the Earthdata user name and password"
            )
    return env["EARTHDATA_USERNAME"], env["EARTHDATA_PASSWORD"]


def login(transport: Transport, username: str, password: str) -> str:
    """The AppEEARS session token (valid for a day). It is kept in memory only."""
    answer = transport.json("POST", "/login", basic=(username, password))
    token = answer.get("token") if isinstance(answer, dict) else None
    if not isinstance(token, str) or not token:
        raise AppEearsError("AppEEARS login did not return a session token")
    return token


def confirm_layers(transport: Transport, spec: SourceSpec) -> dict[str, dict[str, Any]]:
    """Check the product and every layer against the live catalog and return their metadata."""
    listed = {item["ProductAndVersion"] for item in transport.json("GET", "/product")}
    if spec.product not in listed:
        raise AppEearsError(f"{spec.product} is not in the AppEEARS catalog")
    catalog = transport.json("GET", f"/product/{spec.product}")
    confirmed: dict[str, dict[str, Any]] = {}
    for layer in spec.layers:
        if layer.name not in catalog:
            raise AppEearsError(f"{spec.product} has no layer {layer.name}")
        meta = catalog[layer.name]
        scale = meta.get("ScaleFactor")
        confirmed[layer.name] = {
            "units": meta.get("Units") or None,
            "fill_value": meta.get("FillValue"),
            "scale_factor": scale if isinstance(scale, int | float) else None,
            "valid_min": meta.get("ValidMin"),
            "valid_max": meta.get("ValidMax"),
            "data_type": meta.get("DataType"),
        }
    return confirmed


def _us_date(value: str) -> str:
    try:
        return date.fromisoformat(value).strftime("%m-%d-%Y")
    except ValueError:
        raise AppEearsError(f"{value!r} is not a date in YYYY-MM-DD form") from None


def build_request(spec: SourceSpec, bbox: Bbox, start: str, end: str, name: str) -> dict[str, Any]:
    """An AppEEARS `area` task for a box, as GeoTIFF in geographic coordinates."""
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise AppEearsError(
            "The box must be west, south, east, north with west < east and south < north"
        )
    if not (west >= -180 and east <= 180 and south >= -90 and north <= 90):
        raise AppEearsError("The box is outside the range of latitude and longitude")
    start_us, end_us = _us_date(start), _us_date(end)
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise AppEearsError("The start date is after the end date")
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {
        "task_type": "area",
        "task_name": name,
        "params": {
            "dates": [{"startDate": start_us, "endDate": end_us}],
            "layers": [{"product": spec.product, "layer": layer.name} for layer in spec.layers],
            "geo": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                ],
            },
            "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
        },
    }


def request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def submit_or_resume(
    transport: Transport, token: str, request: dict[str, Any], state_dir: Path
) -> str:
    """Submit the task once, then poll it once per call. Returns the id of a `done` task.

    Raises `TaskPending` while it runs, and `AppEearsError` when it failed (its state file is
    removed so the next run submits a new task).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    key = request_key(request)
    state_path = state_dir / f"{key}.json"
    if state_path.is_file():
        task_id = str(json.loads(state_path.read_text(encoding="utf-8"))["task_id"])
    else:
        answer = transport.json("POST", "/task", request, token)
        task_id = str(answer["task_id"])
        state_path.write_text(
            json.dumps({"task_id": task_id, "request_sha256": key}, indent=1) + "\n",
            encoding="utf-8",
        )
    status = str(transport.json("GET", f"/task/{task_id}", token=token).get("status", ""))
    if status == "done":
        return task_id
    if status in FAILED_STATUSES:
        state_path.unlink(missing_ok=True)
        raise AppEearsError(f"AppEEARS task {task_id} ended with status {status}")
    raise TaskPending(task_id, status or "unknown")
