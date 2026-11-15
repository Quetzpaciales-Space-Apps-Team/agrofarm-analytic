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
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any, Protocol

import pandas as pd

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
    usable_classes: tuple[int, ...] = ()  # pixel reliability classes counted as usable


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
        usable_classes=(0, 1),  # good data, marginal data
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
        usable_classes=(0, 1, 2),  # excellent, good, acceptable
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


class StripAuthorizationOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect, but never send the session token to a different host.

    The bundle endpoint answers with a redirect to a CloudFront address that serves the file
    itself. That host needs no token, and the token must not leave the AppEEARS host.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            old_host = urllib.parse.urlsplit(req.full_url).netloc
            if urllib.parse.urlsplit(newurl).netloc != old_host:
                redirected.remove_header("Authorization")
        return redirected


class UrllibTransport:
    """Standard-library transport with a bounded retry on network errors and 5xx answers."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep = sleep
        self._opener = urllib.request.build_opener(StripAuthorizationOnRedirect)

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
                return self._opener.open(request, timeout=120)
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


# --- the bundle -------------------------------------------------------------------------------

MIN_USABLE_FRACTION = 0.5  # a composite is trusted when at least half of its pixels are usable
MIN_SAMPLES_PER_DAY = 6  # SMAP L4 is 3-hourly: a day needs six of its eight samples
PROCESSING_VERSION = "appeears-0.1.0"
RAW_LISTING = "bundle_listing.json"
ATTRIBUTION = (
    "AppEEARS Team, NASA EOSDIS Land Processes DAAC and USGS EROS Center "
    "(https://appeears.earthdatacloud.nasa.gov). Cite the product DOI of the data you use."
)


def _slug(name: str) -> str:
    return name.strip("_").replace("_", "-")


def _bundle_names(spec: SourceSpec) -> list[str]:
    """The files of the bundle the connector reads. Never the request echo or the rasters."""
    prefix = spec.product.replace(".", "-")
    names = [f"{prefix}-Statistics.csv"]
    for layer in spec.layers:
        if layer.role == "quality":
            names.append(f"{prefix}-{_slug(layer.name)}-Statistics-QA.csv")
            names.append(f"{prefix}-{_slug(layer.name)}-lookup.csv")
    return names


def select_bundle_files(listing: dict[str, Any], spec: SourceSpec) -> list[dict[str, Any]]:
    by_name = {item["file_name"]: item for item in listing["files"]}
    chosen = []
    for name in _bundle_names(spec):
        if name not in by_name:
            raise AppEearsError(f"The bundle has no {name}")
        chosen.append(by_name[name])
    return chosen


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AppEearsConnector:
    """`Connector` for one product, one polygon (a box) and one date range."""

    def __init__(
        self,
        spec: SourceSpec,
        bbox: Bbox,
        start: str,
        end: str,
        *,
        transport: Transport,
        token: str,
        state_dir: Path,
        area_name: str,
        min_usable_fraction: float = MIN_USABLE_FRACTION,
        min_samples_per_day: int = MIN_SAMPLES_PER_DAY,
        wait_seconds: float = 0.0,
        poll_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        retrieved_at: str | None = None,
    ) -> None:
        self.spec = spec
        self.dataset = spec.dataset
        key = hashlib.sha256(f"{bbox}".encode()).hexdigest()[:8]
        self.version = f"{spec.product}_{start}_{end}_{key}"
        self.bbox, self.start, self.end, self.area_name = bbox, start, end, area_name
        self.transport, self.token, self.state_dir = transport, token, state_dir
        self.min_usable_fraction = min_usable_fraction
        self.min_samples_per_day = min_samples_per_day
        self.wait_seconds, self.poll_seconds = wait_seconds, poll_seconds
        self.sleep, self.clock = sleep, clock
        self.retrieved_at = retrieved_at or date.today().isoformat()
        self._confirmed: dict[str, dict[str, Any]] = {}
        self._task_id = ""
        self._lookup: dict[str, str] = {}
        self._staging: Path | None = None

    def fetch(self) -> list[Path]:
        """Confirm the ids, submit or resume the task, and download the statistics files.

        Raises `TaskPending` when the task is not done. It waits only if `wait_seconds` is set.
        """
        self._confirmed = confirm_layers(self.transport, self.spec)
        request = build_request(self.spec, self.bbox, self.start, self.end, self.area_name)
        deadline = self.clock() + self.wait_seconds
        while True:
            try:
                self._task_id = submit_or_resume(
                    self.transport, self.token, request, self.state_dir
                )
                break
            except TaskPending:
                if self.clock() >= deadline:
                    raise
                self.sleep(self.poll_seconds)
        listing = self.transport.json("GET", f"/bundle/{self._task_id}", token=self.token)
        chosen = select_bundle_files(listing, self.spec)
        staging = self._staging = Path(tempfile.mkdtemp(prefix="appeears-staging-"))
        files = []
        for item in chosen:
            target = staging / item["file_name"]
            self.transport.download(
                f"/bundle/{self._task_id}/{item['file_id']}", self.token, target
            )
            if _sha256(target) != item["sha256"]:
                raise AppEearsError(
                    f"{item['file_name']}: checksum does not match the bundle listing"
                )
            files.append(target)
        # The listing is kept without the signed file URLs and without the account name.
        kept = {key: listing[key] for key in ("task_id", "bundle_type", "created", "updated")}
        kept["files"] = [
            {key: item[key] for key in ("file_id", "file_name", "file_size", "file_type", "sha256")}
            for item in chosen
        ]
        listing_path = staging / RAW_LISTING
        listing_path.write_text(json.dumps(kept, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return [listing_path, *files]

    def cleanup(self) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None

    # --- cleaning ---

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        by_name = {path.name: path for path in raw_files}
        if any(layer.role == "quality" for layer in self.spec.layers):
            return self._clean_vegetation(by_name)
        return self._clean_soil_moisture(by_name)

    def _read_statistics(self, by_name: dict[str, Path]) -> pd.DataFrame:
        name = f"{self.spec.product.replace('.', '-')}-Statistics.csv"
        table = pd.read_csv(by_name[name])
        if table.empty:
            raise AppEearsError(f"{name} has no rows")
        return table

    def _clean_vegetation(self, by_name: dict[str, Path]) -> pd.DataFrame:
        stats = self._read_statistics(by_name)
        stats["date"] = pd.to_datetime(stats["Date"]).dt.normalize()
        quality = next(layer for layer in self.spec.layers if layer.role == "quality")
        prefix = self.spec.product.replace(".", "-")
        qa = pd.read_csv(by_name[f"{prefix}-{_slug(quality.name)}-Statistics-QA.csv"])
        qa["date"] = pd.to_datetime(qa["Date"], utc=True).dt.tz_convert(None).dt.normalize()
        classes = [column for column in qa.columns if column.isdigit()]
        counts = qa[classes].fillna(0.0)  # a blank count is zero pixels
        qa["pixels_total"] = counts.sum(axis=1)
        qa["pixels_usable"] = counts[
            [str(c) for c in self.spec.usable_classes if str(c) in classes]
        ].sum(axis=1)
        lookup = pd.read_csv(by_name[f"{prefix}-{_slug(quality.name)}-lookup.csv"])
        self._lookup = {str(row["Value"]): str(row["MODLAND"]) for _, row in lookup.iterrows()}

        frame = qa[["date", "pixels_total", "pixels_usable"]].copy()
        for layer in self.spec.layers:
            if layer.role not in {"ndvi", "evi"}:
                continue
            part = stats[stats["Dataset"] == layer.name].set_index("date")
            for source, suffix in (
                ("Mean", "mean"),
                ("Median", "median"),
                ("Standard Deviation", "std"),
            ):
                frame[f"{layer.role}_{suffix}"] = frame["date"].map(part[source]).astype("float64")
        frame["usable_fraction"] = frame["pixels_usable"] / frame["pixels_total"].where(
            frame["pixels_total"] > 0
        )
        ok = frame["usable_fraction"] >= self.min_usable_fraction
        frame["status"] = ok.map({True: "ok", False: "too_cloudy"})
        value_columns = [c for c in frame.columns if c.startswith(("ndvi_", "evi_"))]
        frame.loc[~ok, value_columns] = float("nan")  # no value for a cloudy composite, never zero
        frame.insert(1, "platform", self.spec.platform)
        for column in ("pixels_total", "pixels_usable"):
            frame[column] = frame[column].astype("Int64")
        order = [
            "date",
            "platform",
            *value_columns,
            "pixels_total",
            "pixels_usable",
            "usable_fraction",
            "status",
        ]
        return frame[order].sort_values("date").reset_index(drop=True)

    def _clean_soil_moisture(self, by_name: dict[str, Path]) -> pd.DataFrame:
        stats = self._read_statistics(by_name)
        stats["day"] = pd.to_datetime(stats["Date"].str[:10])
        days = sorted(stats["day"].unique())
        frame = pd.DataFrame({"date": days})
        frame.insert(1, "platform", self.spec.platform)
        pixels = pd.Series(0.0, index=range(len(days)))
        for layer in self.spec.layers:
            meta = self._confirmed.get(layer.name, {})
            low, high = meta.get("valid_min"), meta.get("valid_max")
            part = stats[stats["Dataset"] == layer.name]
            valid = part[part["Count"].gt(0) & part["Mean"].notna()]
            if low is not None and high is not None:
                valid = valid[valid["Mean"].between(low, high)]
            valid = valid.assign(weighted=valid["Mean"] * valid["Count"])
            grouped = valid.groupby("day").agg(
                weighted=("weighted", "sum"),
                count=("Count", "sum"),
                samples=("Mean", "size"),
                pixels=("Count", "max"),
            )
            role = layer.role.removeprefix("soil_moisture_")
            samples = frame["date"].map(grouped["samples"]).fillna(0).astype(int)
            mean = (
                (grouped["weighted"] / grouped["count"])
                .reindex(frame["date"])
                .reset_index(drop=True)
            )
            frame[f"sm_{role}_mean"] = mean.where(samples >= self.min_samples_per_day)
            frame[f"samples_{role}"] = samples
            pixels = pixels.combine(frame["date"].map(grouped["pixels"]).fillna(0), max)
        frame["pixels"] = pixels.astype("Int64")
        enough = (
            frame[[c for c in frame.columns if c.startswith("samples_")]]
            .ge(self.min_samples_per_day)
            .all(axis=1)
        )
        frame["status"] = enough.map({True: "ok", False: "too_few_samples"})
        return frame

    # --- manifest ---

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        table_path = out_dir / f"{self.dataset}.parquet"
        frame.to_parquet(table_path, index=False, engine="pyarrow")
        manifest = self._manifest(frame, table_path, raw_files)
        path = out_dir / f"{self.dataset}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def _manifest(
        self, frame: pd.DataFrame, table_path: Path, raw_files: list[Path]
    ) -> dict[str, Any]:
        spec = self.spec
        statuses = frame["status"].value_counts().to_dict()
        vegetation = any(layer.role == "quality" for layer in spec.layers)
        quality: dict[str, Any]
        if vegetation:
            quality = {
                "usable_classes": list(spec.usable_classes),
                "min_usable_fraction": self.min_usable_fraction,
                "lookup": self._lookup,
                "rule": "A composite has values only when at least min_usable_fraction of its "
                "pixels are in a usable class; otherwise the values are null and the status is "
                "too_cloudy.",
            }
            limitations = [
                "The statistics are computed over all pixels of the polygon, cloudy ones included: "
                "the mean includes cloudy pixels, so a date is trusted only when most pixels "
                "are usable.",
                "Polygon statistics only: no per-pixel values are kept.",
                "The polygon is a box, not a farm boundary.",
            ]
        else:
            quality = {
                "min_samples_per_day": self.min_samples_per_day,
                "valid_range": {
                    name: [meta["valid_min"], meta["valid_max"]]
                    for name, meta in self._confirmed.items()
                },
                "rule": "A day is the count-weighted mean of its valid 3-hourly samples and has "
                "a value only when it has min_samples_per_day of them.",
            }
            limitations = [
                "SMAP L4 is a model product on a 9 km grid; a box of a few tens of kilometres "
                "holds only a few pixels, so it describes the area, not a farm.",
                "Samples outside the valid range of the catalog, or with no pixels, are dropped.",
                "The polygon is a box, not a farm boundary.",
            ]
        west, south, east, north = self.bbox
        return {
            "dataset": self.dataset,
            "file": table_path.name,
            "source": f"NASA AppEEARS ({spec.platform}, {spec.product})",
            "source_version": spec.product,
            "retrieved_at": self.retrieved_at,
            "license": f"{spec.license_note} AppEEARS terms of use apply.",
            "attribution": ATTRIBUTION,
            "date_range": {
                "start": frame["date"].min().strftime("%Y-%m-%d"),
                "end": frame["date"].max().strftime("%Y-%m-%d"),
            },
            "time_step": spec.time_step,
            "spatial_resolution": f"{spec.resolution}; statistics of a polygon",
            "checksum_sha256": _sha256(table_path),
            "source_checksum_sha256": {path.name: _sha256(path) for path in raw_files},
            "processing_version": PROCESSING_VERSION,
            "product": spec.product,
            "platform": spec.platform,
            "task_id": self._task_id,
            "bbox": [west, south, east, north],
            "requested_dates": {"start": self.start, "end": self.end},
            "rows": int(len(frame)),
            "layers": self._confirmed,
            "quality_rule": quality,
            "status_counts": {str(key): int(value) for key, value in statuses.items()},
            "limitations": limitations,
        }
