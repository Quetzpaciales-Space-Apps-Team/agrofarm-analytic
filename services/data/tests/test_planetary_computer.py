"""Planetary Computer NDVI connector: recorded STAC and token fixtures, synthetic rasters with
known answers. Nothing here touches the network."""

import copy
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine, from_origin

from agrofarm_data.cli import main
from agrofarm_data.connectors import planetary_computer as pc
from agrofarm_data.connectors.planetary_computer import (
    PlanetaryComputerError,
    Raster,
    modis_ndvi,
    read_window,
    reflectance_params,
    s2_ndvi,
    scene_stats,
    select_evenly,
)
from agrofarm_data.manifest import validate_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "planetary_computer"
BBOX = (-90.56, 14.58, -90.52, 14.62)
CRS = "EPSG:32615"
GRID = from_origin(700000.0, 1610000.0, 10.0, 10.0)
CLEAR_ID = "S2B_MSIL2A_20250118T162519_R040_T15PYS_20250118T201048"
CLOUDY_ID = "S2A_MSIL2A_20250113T162631_R040_T15PYS_20250113T191949"
TERRA_ID = "MOD13Q1.A2025049.h09v07.061.2025070114631"
AQUA_ID = "MYD13Q1.A2025041.h09v07.061.2025058195852"


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text("utf-8"))
    return data


# SCL of the clear scene: row 0 vegetation, bare, water, unclassified (kept); row 1 shadow,
# medium cloud, high cloud, cirrus; row 2 no data, saturated, snow, dark area (only the last kept).
SCL_CLEAR = np.array(
    [[4, 5, 6, 7], [3, 8, 9, 10], [0, 1, 11, 2], [4, 4, 4, 4]], dtype="uint8"
)


class FakeApi:
    """Answers the three Planetary Computer calls from recorded fixtures."""

    def __init__(self, search: str, collection: str, token: str) -> None:
        self.search = fixture(search)
        self.collection = fixture(collection)
        self.token = fixture(token)
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(self, method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        self.calls.append((method, url, body))
        if url.endswith("/search"):
            return copy.deepcopy(self.search)
        if "/sas/v1/token/" in url:
            return copy.deepcopy(self.token)
        if "/collections/" in url:
            return copy.deepcopy(self.collection)
        raise AssertionError(f"unexpected call {method} {url}")


def s2_api() -> FakeApi:
    return FakeApi(
        "search_sentinel_2_l2a.json", "collection_sentinel_2_l2a.json", "token_sentinel_2_l2a.json"
    )


def modis_api() -> FakeApi:
    return FakeApi(
        "search_modis_13q1_061.json", "collection_modis_13q1_061.json", "token_modis_13q1_061.json"
    )


class FakeReader:
    """Serves synthetic 4 x 4 rasters chosen by item id and asset key found in the href."""

    def __init__(self, arrays: dict[tuple[str, str], NDArray[Any]], crs: str = CRS) -> None:
        self.arrays = arrays
        self.crs = crs
        self.hrefs: list[str] = []

    def __call__(self, href: str, bbox: pc.Bbox, like: Raster | None) -> Raster:
        self.hrefs.append(href)
        for (item_id, asset), data in self.arrays.items():
            if item_id.split(".")[0] in href and asset in href:
                if data is None:
                    raise OSError("HTTP response code: 403")
                return Raster(data, GRID, self.crs)
        raise AssertionError(f"no synthetic raster for {href}")


def s2_arrays() -> dict[tuple[str, str], NDArray[Any]]:
    full = np.full((4, 4), 0, dtype="uint16")
    return {
        (CLEAR_ID, "_B04_"): full + 1500,
        (CLEAR_ID, "_B08_"): full + 4500,
        (CLEAR_ID, "_SCL_"): SCL_CLEAR,
        (CLOUDY_ID, "_B04_"): full + 1500,
        (CLOUDY_ID, "_B08_"): full + 4500,
        (CLOUDY_ID, "_SCL_"): np.full((4, 4), 9, dtype="uint8"),
    }


def run_s2(tmp_path: Path, reader: FakeReader | None = None, **kwargs: Any) -> tuple[Path, Any]:
    api = s2_api()
    out = tmp_path / "out"
    manifest = pc.run_planetary_ndvi(
        "sentinel-2-l2a", BBOX, "2025-01-01", "2025-01-31", out, "test box",
        http=api, reader=reader or FakeReader(s2_arrays()),
        retrieved_at="2025-11-14", **kwargs,
    )  # fmt: skip
    return manifest, api


# --- pixel maths with known answers ------------------------------------------------------


def test_reflectance_offset_is_read_from_the_processing_baseline() -> None:
    item = fixture("search_sentinel_2_l2a.json")["features"][0]
    assert item["properties"]["s2:processing_baseline"] == "05.11"
    scale, offset = reflectance_params(item)
    assert scale == pytest.approx(0.0001)
    assert offset == pytest.approx(-0.1)  # baseline 04.00 and later: DN - 1000


def test_old_baseline_has_no_offset() -> None:
    item = {"properties": {"s2:processing_baseline": "03.01"}, "assets": {}}
    assert reflectance_params(item) == (pytest.approx(0.0001), 0.0)


def test_stac_raster_bands_win_over_the_baseline_rule() -> None:
    item = {
        "properties": {"s2:processing_baseline": "05.11"},
        "assets": {"B04": {"raster:bands": [{"scale": 0.0002, "offset": -0.2}]}},
    }
    assert reflectance_params(item) == (pytest.approx(0.0002), pytest.approx(-0.2))


def test_item_without_baseline_or_scale_is_rejected() -> None:
    with pytest.raises(PlanetaryComputerError, match="processing_baseline"):
        reflectance_params({"properties": {}, "assets": {}})


def test_ndvi_uses_the_offset_and_masks_cloud_classes() -> None:
    b04 = np.full((4, 4), 1500, dtype="uint16")  # (1500 - 1000) / 10000 = 0.05
    b08 = np.full((4, 4), 4500, dtype="uint16")  # (4500 - 1000) / 10000 = 0.35
    ndvi = s2_ndvi(b04, b08, SCL_CLEAR, scale=0.0001, offset=-0.1)
    assert ndvi.dtype == np.float32
    assert np.nanmean(ndvi) == pytest.approx(0.75)  # (0.35 - 0.05) / (0.35 + 0.05)
    # Without the offset the answer would be 0.5: the offset matters.
    plain = s2_ndvi(b04, b08, SCL_CLEAR, scale=0.0001, offset=0.0)
    assert np.nanmean(plain) == pytest.approx(0.5)
    kept = ~np.isnan(ndvi)
    assert kept.sum() == 9
    for cls in (0, 1, 3, 8, 9, 10, 11):
        assert np.isnan(ndvi[SCL_CLEAR == cls]).all(), cls
    for cls in (2, 4, 5, 6, 7):
        assert not np.isnan(ndvi[SCL_CLEAR == cls]).any(), cls


def test_ndvi_is_per_pixel_and_zero_dn_is_no_data() -> None:
    b04 = np.array([[2000, 0], [1200, 1000]], dtype="uint16")
    b08 = np.array([[3000, 5000], [0, 1000]], dtype="uint16")
    scl = np.full((2, 2), 4, dtype="uint8")
    ndvi = s2_ndvi(b04, b08, scl, scale=0.0001, offset=-0.1)
    assert ndvi[0, 0] == pytest.approx((0.2 - 0.1) / (0.2 + 0.1))
    assert np.isnan(ndvi[0, 1])  # B04 no data
    assert np.isnan(ndvi[1, 0])  # B08 no data
    assert np.isnan(ndvi[1, 1])  # both reflectances 0 after the offset: no signal, not 0


def test_dark_pixel_below_the_offset_is_clamped_not_negative() -> None:
    b04 = np.array([[500]], dtype="uint16")  # (500 - 1000) / 10000 < 0
    b08 = np.array([[3000]], dtype="uint16")
    ndvi = s2_ndvi(b04, b08, np.full((1, 1), 4, dtype="uint8"), scale=0.0001, offset=-0.1)
    assert ndvi[0, 0] == pytest.approx(1.0)  # red clamped to 0


def test_modis_scale_fill_and_pixel_reliability() -> None:
    raw = np.array([[5000, -3000, 8000, 2000], [-2000, 9999, 4000, 6000]], dtype="int16")
    reliability = np.array([[0, 0, 1, 2], [0, 3, -1, 1]], dtype="int8")
    ndvi = modis_ndvi(raw, reliability)
    assert ndvi[0, 0] == pytest.approx(0.5)  # 5000 * 0.0001, good
    assert np.isnan(ndvi[0, 1])  # fill value -3000
    assert ndvi[0, 2] == pytest.approx(0.8)  # marginal is kept
    assert np.isnan(ndvi[0, 3])  # snow/ice
    assert ndvi[1, 0] == pytest.approx(-0.2)  # real negative NDVI is a value
    assert np.isnan(ndvi[1, 1])  # cloudy
    assert np.isnan(ndvi[1, 2])  # reliability fill -1
    assert ndvi[1, 3] == pytest.approx(0.6)


def test_scene_stats_null_when_nothing_is_valid() -> None:
    stats = scene_stats(np.full((3, 3), np.nan, dtype="float32"), 0.25)
    assert stats["ndvi_mean"] is None
    assert stats["ndvi_p10"] is None
    assert stats["valid_fraction"] == 0.0
    assert stats["valid_pixels"] == 0
    assert stats["total_pixels"] == 9
    assert stats["status"] == "too_few_valid_pixels"


def test_scene_stats_threshold_is_inclusive_and_never_zero_filled() -> None:
    values = np.full((4, 4), np.nan, dtype="float32")
    values.flat[:4] = [0.2, 0.4, 0.6, 0.8]  # exactly 25 percent valid
    stats = scene_stats(values, 0.25)
    assert stats["status"] == "ok"
    assert stats["ndvi_mean"] == pytest.approx(0.5)
    assert stats["ndvi_p10"] == pytest.approx(0.26)
    assert stats["ndvi_p90"] == pytest.approx(0.74)
    below = scene_stats(values, 0.3)
    assert below["status"] == "too_few_valid_pixels"
    assert below["ndvi_mean"] is None
    assert below["valid_fraction"] == pytest.approx(0.25)  # still reported


# --- search, token, selection -------------------------------------------------------------


def test_search_body_and_user_agent_free_of_personal_data() -> None:
    api = s2_api()
    items = pc.search_items(api, pc.SPECS["sentinel-2-l2a"], BBOX, "2025-01-01", "2025-01-31", 80)
    assert [i["id"] for i in items] == [CLOUDY_ID, CLEAR_ID]  # sorted by date
    method, url, body = api.calls[0]
    assert (method, url) == ("POST", "https://planetarycomputer.microsoft.com/api/stac/v1/search")
    assert body is not None
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["bbox"] == list(BBOX)
    assert body["datetime"] == "2025-01-01T00:00:00Z/2025-01-31T23:59:59Z"
    assert body["query"] == {"eo:cloud_cover": {"lt": 80}}
    assert body["limit"] == 100


def test_modis_search_has_no_cloud_query_and_dates_from_start_datetime() -> None:
    api = modis_api()
    items = pc.search_items(api, pc.SPECS["modis-13q1-061"], BBOX, "2025-01-01", "2025-03-01", 80)
    body = api.calls[0][2]
    assert body is not None
    assert body["collections"] == ["modis-13Q1-061"]  # the id is case sensitive
    assert "query" not in body
    assert [i["id"] for i in items] == [AQUA_ID, TERRA_ID]


def test_search_follows_next_links_and_drops_duplicates() -> None:
    page_one = fixture("search_sentinel_2_l2a.json")
    page_one["features"] = page_one["features"][:1]
    page_one["links"] = [
        {"rel": "next", "method": "POST", "href": "https://x/search", "body": {"token": "next:1"},
         "merge": True}
    ]  # fmt: skip
    page_two = fixture("search_sentinel_2_l2a.json")  # repeats the first item
    bodies: list[dict[str, Any] | None] = []

    def http(method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        bodies.append(body)
        return page_one if len(bodies) == 1 else page_two

    items = pc.search_items(http, pc.SPECS["sentinel-2-l2a"], BBOX, "2025-01-01", "2025-01-31", 100)
    assert len(items) == 2
    assert bodies[1] is not None
    assert bodies[1]["token"] == "next:1"
    assert bodies[1]["collections"] == ["sentinel-2-l2a"]  # merged with the first body


def test_select_evenly_spreads_over_time_and_keeps_all_when_small() -> None:
    items = [{"id": str(n)} for n in range(10)]
    assert [i["id"] for i in select_evenly(items, 4)] == ["0", "3", "6", "9"]
    assert select_evenly(items, 20) == items
    assert select_evenly(items, None) == items
    assert [i["id"] for i in select_evenly(items, 1)] == ["0"]


def test_token_paths_per_collection() -> None:
    api = modis_api()
    token = pc.fetch_token(api, pc.SPECS["modis-13q1-061"])
    assert token == "PLACEHOLDER-SAS-TOKEN"
    assert api.calls[0][1].endswith("/sas/v1/token/modiseuwest/modis-061-cogs")
    api = s2_api()
    pc.fetch_token(api, pc.SPECS["sentinel-2-l2a"])
    assert api.calls[0][1].endswith("/sas/v1/token/sentinel-2-l2a")


def test_unknown_collection_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PlanetaryComputerError, match="collection"):
        pc.run_planetary_ndvi("landsat", BBOX, "2025-01-01", "2025-01-31", tmp_path, "x")


# --- end to end ---------------------------------------------------------------------------


def test_sentinel_series_layers_and_manifest(tmp_path: Path) -> None:
    reader = FakeReader(s2_arrays())
    manifest_path, api = run_s2(tmp_path, reader)
    manifest = validate_manifest(manifest_path)
    out = manifest_path.parent
    table = pd.read_parquet(out / manifest["file"])

    assert list(table.columns[:8]) == [
        "date", "platform", "item_id", "ndvi_mean", "ndvi_p10", "ndvi_p90",
        "valid_fraction", "cloud_cover_item",
    ]  # fmt: skip
    assert len(table) == 2
    table = table.set_index("item_id")
    clear = table.loc[CLEAR_ID]
    assert str(clear["date"]) == "2025-01-18"
    assert clear["platform"] == "Sentinel-2B"
    assert clear["ndvi_mean"] == pytest.approx(0.75)
    assert clear["valid_fraction"] == pytest.approx(9 / 16)
    assert clear["valid_pixels"] == 9
    assert clear["cloud_cover_item"] == pytest.approx(0.990514)
    assert clear["status"] == "ok"
    cloudy = table.loc[CLOUDY_ID]
    assert pd.isna(cloudy["ndvi_mean"])  # all cloud: null, never 0 and never dropped
    assert cloudy["valid_fraction"] == 0.0
    assert cloudy["status"] == "too_few_valid_pixels"

    layer = out / "layers" / f"{CLEAR_ID}.tif"
    with rasterio.open(layer) as tif:
        assert tif.profile["tiled"]
        assert tif.crs.to_string() == CRS
        assert tif.transform == GRID
        assert np.isnan(tif.nodata)
        data = tif.read(1)
    assert np.isnan(data).sum() == 7
    assert np.nanmean(data) == pytest.approx(0.75)
    with rasterio.open(out / "layers" / f"{CLOUDY_ID}.tif") as tif:
        assert np.isnan(tif.read(1)).all()

    assert manifest["stac_collection"] == "sentinel-2-l2a"
    assert manifest["stac_item_ids"] == [CLOUDY_ID, CLEAR_ID]
    assert manifest["date_range"] == {"start": "2025-01-01", "end": "2025-01-31"}
    assert manifest["license"].startswith("proprietary")
    assert "Copernicus" in manifest["attribution"]
    assert "10 m" in manifest["spatial_resolution"]
    assert manifest["scale_factors"]["B04"]["offset"] == pytest.approx(-0.1)
    assert "SCL" in manifest["cloud_mask_rule"]
    assert manifest["cloud_mask_rule"].count("11") >= 1
    assert manifest["min_valid_fraction"] == 0.25
    assert any("team assumption" in text for text in manifest["limitations"])
    assert {entry["item_id"] for entry in manifest["scenes"]} == {CLEAR_ID, CLOUDY_ID}
    assert len(manifest["layers"]) == 2
    assert "api/stac/v1/search" in manifest["source"]
    # The SAS token is signed into hrefs for reading only: it never reaches a stored file.
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".parquet"}:
            assert b"PLACEHOLDER-SAS-TOKEN" not in path.read_bytes(), path
    assert all(h.endswith("?PLACEHOLDER-SAS-TOKEN") for h in reader.hrefs)


def test_rerun_changes_nothing(tmp_path: Path) -> None:
    first, _ = run_s2(tmp_path)
    before = first.read_bytes()
    parquet_before = (first.parent / json.loads(before)["file"]).read_bytes()
    second, _ = run_s2(tmp_path)
    assert second.read_bytes() == before
    assert (second.parent / json.loads(before)["file"]).read_bytes() == parquet_before


def test_raw_inputs_are_stored_once_and_listed_in_the_manifest(tmp_path: Path) -> None:
    manifest_path, _ = run_s2(tmp_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    raw = [p for p in (tmp_path / "out" / "raw").rglob("*") if p.is_file()]
    stored = {p.name for p in raw if not p.name.endswith(".sha256")}
    assert set(manifest["source_checksum_sha256"]) == stored
    assert "stac_items.json" in stored
    assert f"{CLEAR_ID}__B04.tif" in stored


def test_unreadable_scene_stays_in_the_table_with_a_status(tmp_path: Path) -> None:
    arrays = s2_arrays()
    arrays[(CLEAR_ID, "_B08_")] = None  # type: ignore[assignment]
    manifest_path, _ = run_s2(tmp_path, FakeReader(arrays))
    manifest = json.loads(manifest_path.read_text("utf-8"))
    table = pd.read_parquet(manifest_path.parent / manifest["file"]).set_index("item_id")
    assert len(table) == 2
    assert table.loc[CLEAR_ID, "status"] == "read_failed"
    assert pd.isna(table.loc[CLEAR_ID, "ndvi_mean"])
    assert any(CLEAR_ID in text and "403" in text for text in manifest["warnings"])
    assert not (manifest_path.parent / "layers" / f"{CLEAR_ID}.tif").exists()


def test_max_items_thins_the_scenes(tmp_path: Path) -> None:
    manifest_path, _ = run_s2(tmp_path, max_items=1)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["stac_item_ids"] == [CLOUDY_ID]
    assert manifest["scenes_found"] == 2


def test_empty_search_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    api = s2_api()
    api.search["features"] = []
    with pytest.raises(PlanetaryComputerError, match="no scenes"):
        pc.run_planetary_ndvi(
            "sentinel-2-l2a", BBOX, "2025-01-01", "2025-01-31", tmp_path, "x",
            http=api, reader=FakeReader({}),
        )  # fmt: skip


def test_modis_series_uses_native_grid_and_composite_dates(tmp_path: Path) -> None:
    sinusoidal = "ESRI:54008"
    good = np.full((4, 4), 5000, dtype="int16")
    good[0, 0] = -3000  # fill
    reliability = np.zeros((4, 4), dtype="int8")
    reliability[1, :] = 3  # cloudy row
    arrays: dict[tuple[str, str], NDArray[Any]] = {}
    for item_id in (TERRA_ID, AQUA_ID):
        arrays[(item_id, "_NDVI")] = good
        arrays[(item_id, "pixel_reliability")] = reliability
    api = modis_api()
    out = tmp_path / "modis"
    manifest_path = pc.run_planetary_ndvi(
        "modis-13q1-061", BBOX, "2025-01-01", "2025-03-31", out, "test box",
        http=api, reader=FakeReader(arrays, crs=sinusoidal), retrieved_at="2025-11-14",
    )  # fmt: skip
    manifest = validate_manifest(manifest_path)
    table = pd.read_parquet(out / manifest["file"]).set_index("item_id")
    terra = table.loc[TERRA_ID]
    assert str(terra["date"]) == "2025-02-18"  # start of the 16-day composite
    assert str(terra["date_end"]) == "2025-03-05"
    assert terra["platform"] == "MODIS Terra"
    assert table.loc[AQUA_ID, "platform"] == "MODIS Aqua"
    assert terra["ndvi_mean"] == pytest.approx(0.5)
    assert terra["valid_fraction"] == pytest.approx(11 / 16)
    assert pd.isna(terra["cloud_cover_item"])
    assert manifest["scale_factors"]["250m_16_days_NDVI"]["scale"] == 0.0001
    assert manifest["scale_factors"]["250m_16_days_NDVI"]["fill_value"] == -3000
    assert "pixel reliability" in manifest["cloud_mask_rule"]
    assert "250 m" in manifest["spatial_resolution"]
    with rasterio.open(out / "layers" / f"{TERRA_ID}.tif") as tif:
        assert tif.crs == rasterio.crs.CRS.from_string(sinusoidal)
        assert np.isnan(tif.read(1)).sum() == 5


# --- the real HTTP and raster functions, offline ------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_http_json_retries_and_sends_a_generic_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        seen.append(request)
        if len(seen) < 3:
            raise urllib.error.URLError("temporary")
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pauses: list[float] = []
    result = pc.http_json("POST", "https://example.org/search", {"a": 1}, sleep=pauses.append)
    assert result == {"ok": True}
    assert len(seen) == 3
    assert len(pauses) == 2
    request = seen[-1]
    assert request.get_header("User-agent") == "agrofarm-data-connector/0.1"
    assert json.loads(request.data) == {"a": 1}
    assert request.get_method() == "POST"


def test_http_json_gives_up_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(request: Any, timeout: float) -> FakeResponse:
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(PlanetaryComputerError, match="3 attempts"):
        pc.http_json("GET", "https://example.org/x", None, sleep=lambda _: None)


def write_tif(path: Path, data: NDArray[Any], transform: Affine, crs: str) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype=str(data.dtype), crs=crs, transform=transform,
    ) as out:  # fmt: skip
        out.write(data, 1)


def test_read_window_crops_to_the_box_and_resamples_the_20m_layer_by_nearest(
    tmp_path: Path,
) -> None:
    from rasterio.warp import transform_bounds

    west, south, east, north = transform_bounds("EPSG:4326", CRS, *BBOX)
    # A 10 m band and a 20 m class map that cover the box with a margin.
    left = np.floor(west / 100) * 100 - 100
    top = np.ceil(north / 100) * 100 + 100
    size = 1000  # 10 km of 10 m pixels is more than the 4 km box
    band = np.arange(size * size, dtype="uint16").reshape(size, size) % 60000
    classes = (np.arange(size * size // 4) % 11).astype("uint8").reshape(size // 2, size // 2)
    write_tif(tmp_path / "band.tif", band, from_origin(left, top, 10, 10), CRS)
    write_tif(tmp_path / "scl.tif", classes, from_origin(left, top, 20, 20), CRS)

    reference = read_window(str(tmp_path / "band.tif"), BBOX, None)
    assert reference.crs == CRS
    assert reference.transform.a == 10.0
    height, width = reference.data.shape
    assert 380 <= width <= 450  # about 4 km of 10 m pixels, whole pixels
    bounds = rasterio.transform.array_bounds(height, width, reference.transform)
    assert bounds[0] <= west and bounds[2] >= east and bounds[1] <= south and bounds[3] >= north

    coarse = read_window(str(tmp_path / "scl.tif"), BBOX, reference)
    assert coarse.data.shape == reference.data.shape
    assert coarse.transform == reference.transform
    assert coarse.data.dtype == np.uint8
    assert set(np.unique(coarse.data)) <= set(range(11))  # nearest: no blended classes
    assert (coarse.data[:, 0::2][:, :-1] == coarse.data[:, 1::2][:, : coarse.data[:, 1::2].shape[1]]).mean() > 0.4


# --- command line -------------------------------------------------------------------------


def test_command_line_runs_with_injected_http_and_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli"
    code = main(
        ["planetary-ndvi", "--collection", "sentinel-2-l2a", "--bbox", "-90.56", "14.58",
         "-90.52", "14.62", "--start", "2025-01-01", "--end", "2025-01-31", "--out", str(out),
         "--area-name", "test box", "--max-items", "5"],
        http=s2_api(), reader=FakeReader(s2_arrays()),
    )  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "2 scenes" in text
    assert "1 with NDVI" in text
    assert (out / "ndvi_sentinel_2_l2a.manifest.json").is_file()


def test_command_line_rejects_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["planetary-ndvi", "--bbox", "-90.56", "14.58", "-90.52", "14.62", "--out", "x",
            "--area-name", "a"]  # fmt: skip
    with pytest.raises(SystemExit):
        main([*base, "--collection", "landsat", "--start", "2025-01-01", "--end", "2025-01-31"])
    with pytest.raises(SystemExit):
        main([*base, "--collection", "sentinel-2-l2a", "--start", "2025-02-01",
              "--end", "2025-01-31"])  # fmt: skip
    with pytest.raises(SystemExit):
        main([*base[:2], "-90.52", "14.58", "-90.56", "14.62", *base[6:],
              "--collection", "sentinel-2-l2a", "--start", "2025-01-01", "--end", "2025-01-31"])
    assert "error" in capsys.readouterr().err.lower()
