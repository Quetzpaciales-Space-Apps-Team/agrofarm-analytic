"""Microsoft Planetary Computer NDVI connector: Sentinel-2 L2A and MODIS 13Q1 (DS-031).

One code path, two collections (`SPECS`). For a small box and a date range it searches the public
STAC API, signs the asset hrefs with an anonymous SAS token, reads only a window of each
cloud-optimized GeoTIFF, computes NDVI with a cloud mask and writes:

- a Parquet series, one row per scene (mean, p10 and p90 of NDVI over valid pixels, the share of
  valid pixels, the item cloud cover),
- one tiled GeoTIFF of NDVI per scene in `layers/` (nodata NaN, native grid of the source),
- a manifest.

The standard library talks to the two public endpoints (`http_json`), and rasterio reads the
windows (`read_window`). Both are injectable, so tests never use the network. The raw windows
and the STAC response are kept in the write-once raw store and the clean outputs are computed
from those stored copies (`run_job`). The SAS token is never written to any file.
"""

import hashlib
import json
import math
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from agrofarm_data import __version__
from agrofarm_data.connector import run_job

Bbox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)
HttpJson = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]

API = "https://planetarycomputer.microsoft.com/api"
SEARCH_URL = f"{API}/stac/v1/search"
USER_AGENT = "agrofarm-data-connector/0.1"
ATTEMPTS = 3
PAGE_LIMIT = 100
MAX_PAGES = 20
MIN_VALID_FRACTION = 0.25  # team assumption: below this share of valid pixels the NDVI is null
RAW_ITEMS = "stac_items.json"

# Sentinel-2 scene classification (SCL) classes that are never used for NDVI.
SCL_MASKED = (0, 1, 3, 8, 9, 10, 11)
SCL_RULE = (
    "Pixels are dropped when the SCL class is 0 (no data), 1 (saturated or defective), "
    "3 (cloud shadow), 8 (cloud medium probability), 9 (cloud high probability), 10 (thin cirrus) "
    "or 11 (snow or ice: not a vegetation signal, and practically absent at this latitude), or "
    "when B04 or B08 is 0 (no data), or when B04 + B08 is not positive. Classes 2 (dark area), "
    "4, 5, 6 (water), and 7 (unclassified, low probability cloud) are kept."
)
MODIS_RULE = (
    "Pixels are kept when the pixel reliability layer is 0 (good) or 1 (marginal). Reliability 2 "
    "(snow or ice), 3 (cloudy) and the fill value are dropped, and so is an NDVI of -3000 (fill)."
)
MODIS_SCALE = 0.0001
MODIS_FILL = -3000
MODIS_KEEP = (0, 1)
S2_SCALE = 0.0001


class PlanetaryComputerError(ValueError):
    """The API answered with something unusable, or the request cannot be served."""


class Raster(NamedTuple):
    data: NDArray[Any]
    transform: Affine
    crs: str


RasterReader = Callable[[str, Bbox, "Raster | None"], Raster]


@dataclass(frozen=True)
class CollectionSpec:
    """Everything that differs between collections. The code path is shared."""

    cli_id: str  # lowercase id used on the command line
    stac_id: str  # the exact STAC collection id (case sensitive)
    dataset: str
    source_version: str
    token_path: str  # path under /sas/v1/token/ that signs the COG container
    assets: tuple[str, ...]  # first asset defines the pixel grid
    cloud_query: bool  # the items carry eo:cloud_cover
    resolution: str
    attribution: str
    license_note: str
    cloud_mask_rule: str


SPECS: dict[str, CollectionSpec] = {
    "sentinel-2-l2a": CollectionSpec(
        cli_id="sentinel-2-l2a",
        stac_id="sentinel-2-l2a",
        dataset="ndvi_sentinel2_l2a",
        source_version="Level-2A",
        token_path="sentinel-2-l2a",  # noqa: S106
        assets=("B04", "B08", "SCL"),
        cloud_query=True,
        resolution="10 m (B04, B08); the 20 m SCL is resampled to the 10 m grid by nearest "
        "neighbour; native UTM grid of the scene",
        attribution="Contains modified Copernicus Sentinel data, processed by ESA; accessed "
        "through Microsoft Planetary Computer.",
        license_note="Copernicus Sentinel data are free, full and open (Commission Delegated "
        "Regulation (EU) No 1159/2013). The collection links the Sentinel data terms.",
        cloud_mask_rule=SCL_RULE,
    ),
    "modis-13q1-061": CollectionSpec(
        cli_id="modis-13q1-061",
        stac_id="modis-13Q1-061",
        dataset="ndvi_modis_13q1_061",
        source_version="061",
        token_path="modiseuwest/modis-061-cogs",  # noqa: S106
        assets=("250m_16_days_NDVI", "250m_16_days_pixel_reliability"),
        cloud_query=False,
        resolution="250 m (about 231.66 m), native MODIS sinusoidal grid; 16-day composites",
        attribution="MOD13Q1 and MYD13Q1 Version 061, NASA LP DAAC at the USGS EROS Center; "
        "accessed through Microsoft Planetary Computer.",
        license_note="The collection metadata gives no license link. NASA Earth science data "
        "(MODIS) are open under the NASA EOSDIS data and information policy; the team confirms "
        "the citation wording before publication.",
        cloud_mask_rule=MODIS_RULE,
    ),
}


# --- HTTP and raster access (the two injectable seams) ------------------------------------


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Call a public JSON endpoint with the standard library. Three attempts, then an error."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                parsed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < ATTEMPTS - 1:
                sleep(2.0 * (attempt + 1))
            continue
        if not isinstance(parsed, dict):
            raise PlanetaryComputerError(f"{url} did not return a JSON object")
        return parsed
    raise PlanetaryComputerError(f"{method} {url} failed after {ATTEMPTS} attempts: {last}")


def read_window(href: str, bbox: Bbox, like: Raster | None) -> Raster:
    """Read band 1 of a (remote) GeoTIFF for the box.

    Without `like` the window is the whole pixels that cover the box. With `like` the data are read
    for exactly that grid and resampled by nearest neighbour (used for the 20 m SCL).
    """
    with rasterio.open(href) as source:
        crs = source.crs.to_string()
        if like is None:
            west, south, east, north = transform_bounds("EPSG:4326", source.crs, *bbox, 21)
            window = from_bounds(west, south, east, north, source.transform)
            col0, row0 = math.floor(window.col_off), math.floor(window.row_off)
            col1 = math.ceil(window.col_off + window.width)
            row1 = math.ceil(window.row_off + window.height)
            full = Window(0, 0, source.width, source.height)
            window = Window(col0, row0, col1 - col0, row1 - row0).intersection(full)
            data = source.read(1, window=window)
            return Raster(data, source.window_transform(window), crs)
        height, width = like.data.shape
        bounds = array_bounds(height, width, like.transform)
        window = from_bounds(*bounds, source.transform)
        data = source.read(
            1, window=window, out_shape=(height, width), resampling=Resampling.nearest
        )
        return Raster(data, like.transform, crs)


# --- search ---------------------------------------------------------------------------------


def _item_date(item: dict[str, Any]) -> str:
    props = item["properties"]
    return str(props.get("datetime") or props.get("start_datetime") or "")


def search_items(
    http: HttpJson,
    spec: CollectionSpec,
    bbox: Bbox,
    start: str,
    end: str,
    max_cloud: float,
) -> list[dict[str, Any]]:
    """All STAC items of the collection for the box and dates, oldest first, no duplicates."""
    body: dict[str, Any] = {
        "collections": [spec.stac_id],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": PAGE_LIMIT,
        "sortby": [{"field": "datetime", "direction": "asc"}],
    }
    if spec.cloud_query:
        body["query"] = {"eo:cloud_cover": {"lt": max_cloud}}
    items: dict[str, dict[str, Any]] = {}
    method, url, request = "POST", SEARCH_URL, body
    for _ in range(MAX_PAGES):
        page = http(method, url, request)
        for item in page.get("features", []):
            items.setdefault(item["id"], item)
        link = next((x for x in page.get("links", []) if x.get("rel") == "next"), None)
        if link is None:
            break
        method, url = link.get("method", "POST"), link["href"]
        request = {**request, **link["body"]} if link.get("merge") else link.get("body")
    return sorted(items.values(), key=_item_date)


def select_evenly(items: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    """Keep `count` items spread evenly over the list (first and last included)."""
    if count is None or count >= len(items):
        return items
    if count <= 1:
        return items[:1]
    last = len(items) - 1
    picks = sorted({round(i * last / (count - 1)) for i in range(count)})
    return [items[i] for i in picks]


def fetch_token(http: HttpJson, spec: CollectionSpec) -> str:
    answer = http("GET", f"{API}/sas/v1/token/{spec.token_path}", None)
    token = answer.get("token")
    if not isinstance(token, str) or not token:
        raise PlanetaryComputerError(f"No SAS token returned for {spec.token_path}")
    return token


# --- pixel maths ------------------------------------------------------------------------------


def reflectance_params(item: dict[str, Any]) -> tuple[float, float]:
    """Return (scale, offset) with reflectance = DN * scale + offset for a Sentinel-2 item.

    The STAC `raster:bands` of the B04 asset win when present. The Planetary Computer items do not
    carry them (checked against the live API), so the processing baseline decides: from 04.00 ESA
    adds an offset of 1000 DN to the stored values, which is -0.1 in reflectance.
    """
    bands = item.get("assets", {}).get("B04", {}).get("raster:bands")
    if bands and "scale" in bands[0]:
        return float(bands[0]["scale"]), float(bands[0].get("offset", 0.0))
    baseline = item.get("properties", {}).get("s2:processing_baseline")
    if not baseline:
        raise PlanetaryComputerError(
            f"{item.get('id')}: no raster:bands and no s2:processing_baseline, "
            "the reflectance offset is unknown"
        )
    major = int(str(baseline).split(".")[0])
    return S2_SCALE, (-1000 * S2_SCALE if major >= 4 else 0.0)


def s2_ndvi(
    b04: NDArray[Any], b08: NDArray[Any], scl: NDArray[Any], *, scale: float, offset: float
) -> NDArray[np.float32]:
    """NDVI = (B08 - B04) / (B08 + B04) on reflectance, NaN where the pixel is not usable."""
    red = np.clip(b04.astype("float64") * scale + offset, 0.0, None)
    nir = np.clip(b08.astype("float64") * scale + offset, 0.0, None)
    total = nir + red
    usable = (b04 > 0) & (b08 > 0) & ~np.isin(scl, SCL_MASKED) & (total > 0)
    ndvi = np.full(b04.shape, np.nan, dtype="float64")
    np.divide(nir - red, total, out=ndvi, where=usable)
    return ndvi.astype("float32")


def modis_ndvi(raw: NDArray[Any], reliability: NDArray[Any]) -> NDArray[np.float32]:
    """MODIS NDVI times 0.0001, NaN for fill values and for reliability other than 0 or 1."""
    usable = np.isin(reliability, MODIS_KEEP) & (raw != MODIS_FILL)
    ndvi = np.where(usable, raw.astype("float64") * MODIS_SCALE, np.nan)
    return ndvi.astype("float32")


def scene_stats(ndvi: NDArray[np.float32], min_valid_fraction: float) -> dict[str, Any]:
    """Series values of one scene. Too little valid area gives null NDVI, never 0."""
    valid = ndvi[~np.isnan(ndvi)]
    fraction = float(valid.size / ndvi.size)
    if valid.size == 0 or fraction < min_valid_fraction:
        mean = p10 = p90 = None
        status = "too_few_valid_pixels"
    else:
        mean, p10, p90 = (
            float(valid.mean()),
            float(np.percentile(valid, 10)),
            float(np.percentile(valid, 90)),
        )
        status = "ok"
    return {
        "ndvi_mean": mean, "ndvi_p10": p10, "ndvi_p90": p90,
        "valid_fraction": round(fraction, 6), "valid_pixels": int(valid.size),
        "total_pixels": int(ndvi.size), "status": status,
    }  # fmt: skip


# --- the connector --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tif(path: Path, data: NDArray[Any], transform: Affine, crs: str, *, tiled: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
        "dtype": str(data.dtype), "crs": crs, "transform": transform, "compress": "deflate",
    }  # fmt: skip
    if tiled:
        profile.update(nodata=float("nan"), tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as out:
        out.write(data, 1)


def _read_tif(path: Path) -> Raster:
    with rasterio.open(path) as source:
        return Raster(source.read(1), source.transform, source.crs.to_string())


def _platform(spec: CollectionSpec, item: dict[str, Any]) -> str:
    if spec.cli_id == "sentinel-2-l2a":
        return str(item["properties"].get("platform", "Sentinel-2"))
    return "MODIS Terra" if item["id"].startswith("MOD") else "MODIS Aqua"


def _dates(item: dict[str, Any]) -> tuple[date, date]:
    props = item["properties"]
    start = date.fromisoformat(_item_date(item)[:10])
    end = date.fromisoformat(str(props.get("end_datetime") or _item_date(item))[:10])
    return start, end


class PlanetaryComputerConnector:
    """`Connector` for one collection and one box. See the module docstring."""

    def __init__(
        self,
        spec: CollectionSpec,
        bbox: Bbox,
        start: str,
        end: str,
        area_name: str,
        *,
        max_items: int | None = None,
        max_cloud: float = 100.0,
        min_valid_fraction: float = MIN_VALID_FRACTION,
        http: HttpJson = http_json,
        reader: RasterReader = read_window,
        retrieved_at: str | None = None,
    ) -> None:
        self.spec = spec
        self.dataset = spec.dataset
        key = f"{bbox}{max_items}{max_cloud}".encode()
        self.version = f"{start}_{end}_{hashlib.sha256(key).hexdigest()[:8]}"
        self.bbox, self.start, self.end, self.area_name = bbox, start, end, area_name
        self.max_items, self.max_cloud = max_items, max_cloud
        self.min_valid_fraction = min_valid_fraction
        self.http, self.reader = http, reader
        self.retrieved_at = retrieved_at or date.today().isoformat()
        self._layers: dict[str, tuple[NDArray[np.float32], Affine, str]] = {}
        self._params: dict[str, tuple[float, float]] = {}
        self._meta: dict[str, Any] = {}
        self._staging: Path | None = None

    @staticmethod
    def raw_name(item_id: str, asset: str) -> str:
        return f"{item_id}__{asset}.tif"

    def fetch(self) -> list[Path]:
        """Search, sign and read one window per asset and scene into a staging folder."""
        found = search_items(self.http, self.spec, self.bbox, self.start, self.end, self.max_cloud)
        if not found:
            raise PlanetaryComputerError(
                f"{self.spec.stac_id}: no scenes for {self.bbox} "
                f"between {self.start} and {self.end}"
            )
        chosen = select_evenly(found, self.max_items)
        token = fetch_token(self.http, self.spec)
        collection = self.http("GET", f"{API}/stac/v1/collections/{self.spec.stac_id}", None)
        links = [x["href"] for x in collection.get("links", []) if x.get("rel") == "license"]
        staging = self._staging = Path(tempfile.mkdtemp(prefix="pc-staging-"))
        files: list[Path] = []
        errors: dict[str, str] = {}
        for item in chosen:
            try:
                rasters = self._read_scene(item, token)
            except (OSError, ValueError, KeyError, RasterioError) as exc:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
                continue
            for asset, raster in rasters.items():
                path = staging / self.raw_name(item["id"], asset)
                _write_tif(path, raster.data, raster.transform, raster.crs, tiled=False)
                files.append(path)
        meta = {
            "collection": self.spec.stac_id,
            "collection_license": collection.get("license"),
            "collection_license_links": links,
            "collection_title": collection.get("title"),
            "scenes_found": len(found),
            "read_errors": errors,
            "items": chosen,
        }
        listing = staging / RAW_ITEMS
        listing.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return [listing, *files]

    def cleanup(self) -> None:
        """Remove the staging folder. The raw store keeps the copies."""
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None

    def _read_scene(self, item: dict[str, Any], token: str) -> dict[str, Raster]:
        rasters: dict[str, Raster] = {}
        reference: Raster | None = None
        for asset in self.spec.assets:
            href = f"{item['assets'][asset]['href']}?{token}"
            raster = self.reader(href, self.bbox, reference)
            reference = reference or raster
            rasters[asset] = raster
        return rasters

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        """Compute NDVI per scene from the stored raw windows. One row per scene, none dropped."""
        by_name = {path.name: path for path in raw_files}
        meta = json.loads(by_name[RAW_ITEMS].read_text("utf-8"))
        self._meta = meta
        rows = []
        for item in meta["items"]:
            item_id = item["id"]
            first, last = _dates(item)
            row: dict[str, Any] = {
                "date": first, "date_end": last, "platform": _platform(self.spec, item),
                "item_id": item_id, "ndvi_mean": None, "ndvi_p10": None, "ndvi_p90": None,
                "valid_fraction": None,
                "cloud_cover_item": item["properties"].get("eo:cloud_cover")
                if self.spec.cloud_query
                else None,
                "valid_pixels": None, "total_pixels": None, "status": "read_failed",
            }  # fmt: skip
            if item_id not in meta["read_errors"]:
                ndvi, transform, crs = self._scene_ndvi(item, by_name)
                self._layers[item_id] = (ndvi, transform, crs)
                row.update(scene_stats(ndvi, self.min_valid_fraction))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.insert(len(frame.columns) - 1, "date_end", frame.pop("date_end"))
        for column in ("valid_pixels", "total_pixels"):
            frame[column] = frame[column].astype("Int64")
        for column in ("ndvi_mean", "ndvi_p10", "ndvi_p90", "valid_fraction", "cloud_cover_item"):
            frame[column] = frame[column].astype("float64")
        return frame

    def _scene_ndvi(
        self, item: dict[str, Any], by_name: dict[str, Path]
    ) -> tuple[NDArray[np.float32], Affine, str]:
        grids = {a: _read_tif(by_name[self.raw_name(item["id"], a)]) for a in self.spec.assets}
        first = grids[self.spec.assets[0]]
        if self.spec.cli_id == "sentinel-2-l2a":
            scale, offset = reflectance_params(item)
            self._params[item["id"]] = (scale, offset)
            ndvi = s2_ndvi(
                grids["B04"].data, grids["B08"].data, grids["SCL"].data, scale=scale, offset=offset
            )
        else:
            ndvi = modis_ndvi(*(grids[a].data for a in self.spec.assets))
        return ndvi, first.transform, first.crs

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        table_path = out_dir / f"{self.dataset}.parquet"
        frame.to_parquet(table_path, index=False, engine="pyarrow")
        layers = []
        for item_id, (ndvi, transform, crs) in self._layers.items():
            target = out_dir / "layers" / f"{item_id}.tif"
            _write_tif(target, ndvi, transform, crs, tiled=True)
            layers.append({"file": f"layers/{target.name}", "item_id": item_id,
                           "checksum_sha256": _sha256(target)})  # fmt: skip
        manifest = self._manifest(frame, table_path, layers, raw_files)
        path = out_dir / f"{self.dataset}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def _scale_factors(self) -> dict[str, Any]:
        if self.spec.cli_id != "sentinel-2-l2a":
            return {
                "250m_16_days_NDVI": {"scale": MODIS_SCALE, "offset": 0.0, "fill_value": MODIS_FILL,
                                      "source": "STAC raster:bands scale of the asset"},
                "250m_16_days_pixel_reliability": {"kept_values": list(MODIS_KEEP)},
            }  # fmt: skip
        variants = sorted(set(self._params.values()))
        entry: dict[str, Any] = {
            "scale": variants[0][0] if variants else S2_SCALE,
            "offset": variants[0][1] if variants else 0.0,
            "formula": "reflectance = DN * scale + offset, clamped at 0",
            "source": "STAC raster:bands when present, else the s2:processing_baseline rule "
            "(offset -0.1 from baseline 04.00)",
        }
        if len(variants) > 1:
            entry["variants"] = [{"scale": s, "offset": o} for s, o in variants]
        return {"B04": entry, "B08": dict(entry)}

    def _manifest(
        self,
        frame: pd.DataFrame,
        table_path: Path,
        layers: list[dict[str, Any]],
        raw_files: list[Path],
    ) -> dict[str, Any]:
        meta = self._meta
        spec = self.spec
        west, south, east, north = self.bbox
        license_text = (
            f"{meta['collection_license']} (Planetary Computer collection metadata). "
            f"{spec.license_note}"
        )
        if meta["collection_license_links"]:
            license_text += f" License link: {meta['collection_license_links'][0]}"
        layer_by_item = {x["item_id"]: x for x in layers}
        scenes = []
        for row in frame.to_dict("records"):
            entry = {
                "item_id": row["item_id"], "date": str(row["date"]), "status": row["status"],
                "valid_fraction": None if pd.isna(row["valid_fraction"]) else row["valid_fraction"],
            }  # fmt: skip
            if row["item_id"] in layer_by_item:
                entry["layer"] = layer_by_item[row["item_id"]]["file"]
            if row["item_id"] in self._params:
                entry["reflectance_scale_offset"] = list(self._params[row["item_id"]])
            scenes.append(entry)
        warnings = [f"{k}: {v}" for k, v in sorted(meta["read_errors"].items())]
        limitations = [
            f"The valid pixel share of a scene must be at least {self.min_valid_fraction:.0%}; "
            "below it ndvi_mean, ndvi_p10 and ndvi_p90 are null and the row stays in the table "
            "(status too_few_valid_pixels). This threshold is a team assumption.",
            "The box is provisional until the farm polygon is decided (DEC-003).",
            "Scenes are not gap-filled or composited. Mean NDVI of a scene covers only its "
            "valid pixels, so the valid share must be read together with the mean.",
            f"The collection license reads {meta['collection_license']!r} in the Planetary "
            "Computer metadata. Check the terms of use of the source before publication.",
        ]
        if spec.cli_id == "sentinel-2-l2a":
            limitations.append(
                "cloud_cover_item is the cloud share of the whole tile, not of the box. "
                "valid_fraction is the cloud-free share of the box. Search filter on item cloud "
                f"cover: below {self.max_cloud}%."
            )
        else:
            limitations.append(
                "The collection holds MOD13Q1 (Terra) and MYD13Q1 (Aqua) composites, each over 16 "
                "days; date is the first day of the composite and date_end the last. A composite "
                "picks the best observation of its period, so one value is not a single day. "
                "The cloud information is the pixel reliability layer, so cloud_cover_item is "
                "null (a few "
                "items carry an eo:cloud_cover of 0.0 that contradicts their pixel reliability)."
            )
        return {
            "dataset": self.dataset,
            "file": table_path.name,
            "source": f"Microsoft Planetary Computer STAC API ({SEARCH_URL}), collection "
            f"{spec.stac_id}; assets read as cloud-optimized GeoTIFF windows over HTTPS",
            "source_version": spec.source_version,
            "retrieved_at": self.retrieved_at,
            "license": license_text,
            "attribution": spec.attribution,
            "area_name": self.area_name,
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "date_range": {"start": self.start, "end": self.end},
            "time_step": "per scene" if spec.cli_id == "sentinel-2-l2a" else "16-day composite",
            "spatial_resolution": spec.resolution,
            "crs": "native grid of the source assets (see each layer)",
            "rows": int(len(frame)),
            "stac_collection": spec.stac_id,
            "stac_item_ids": [x["id"] for x in meta["items"]],
            "scenes_found": meta["scenes_found"],
            "assets": list(spec.assets),
            "search": {
                "max_items": self.max_items,
                "max_cloud_cover": self.max_cloud,
                "endpoint": SEARCH_URL,
            },  # fmt: skip
            "scale_factors": self._scale_factors(),
            "cloud_mask_rule": spec.cloud_mask_rule,
            "min_valid_fraction": self.min_valid_fraction,
            "missing_value": "null in the table, NaN in the GeoTIFFs",
            "limitations": limitations,
            "warnings": warnings,
            "scenes": scenes,
            "layers": layers,
            "source_checksum_sha256": {p.name: _sha256(p) for p in sorted(raw_files)},
            "checksum_sha256": _sha256(table_path),
            "processing_version": __version__,
        }


def run_planetary_ndvi(
    collection: str,
    bbox: Bbox,
    start: str,
    end: str,
    out_dir: Path,
    area_name: str,
    *,
    max_items: int | None = None,
    max_cloud: float = 100.0,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    raw_dir: Path | None = None,
    http: HttpJson = http_json,
    reader: RasterReader = read_window,
    retrieved_at: str | None = None,
) -> Path:
    """Run the connector end to end and return the manifest path."""
    spec = SPECS.get(collection)
    if spec is None:
        raise PlanetaryComputerError(
            f"Unknown collection {collection!r}; use one of {sorted(SPECS)}"
        )
    connector = PlanetaryComputerConnector(
        spec, bbox, start, end, area_name, max_items=max_items, max_cloud=max_cloud,
        min_valid_fraction=min_valid_fraction, http=http, reader=reader, retrieved_at=retrieved_at,
    )  # fmt: skip
    try:
        return run_job(connector, raw_dir or out_dir / "raw", out_dir)
    finally:
        connector.cleanup()
