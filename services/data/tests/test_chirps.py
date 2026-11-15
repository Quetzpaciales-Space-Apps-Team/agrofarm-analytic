"""CHIRPS v3 daily RNL connector uses a recorded GeoTIFF crop, never the network."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from agrofarm_data.connector import run_job
from agrofarm_data.connectors.chirps import (
    ChirpsConnector,
    ChirpsError,
    _read_remote_window,
)

FIXTURE = Path(__file__).parent / "fixtures/chirps/chirps-v3.0.rnl.2024.01.01.tif"
BOX = (-90.60, 14.40, -90.40, 14.60)
DAY = date(2024, 1, 1)


def test_remote_reader_returns_complete_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = rasterio.open
    with real_open(FIXTURE) as fixture:
        monkeypatch.setattr(
            "agrofarm_data.connectors.chirps.rasterio.open",
            lambda _path: fixture,
        )
        result = _read_remote_window("https://chirps.example/day.tif", BOX)
    with rasterio.io.MemoryFile(result) as memory, memory.open() as output:
        assert output.shape == (4, 4)
        assert output.transform.c == pytest.approx(-90.6)
        assert output.transform.f == pytest.approx(14.6)
        assert output.read(1)[1, 0] == pytest.approx(0.00167224, abs=1e-7)


def test_remote_reader_rejects_area_outside_source(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = rasterio.open
    with real_open(FIXTURE) as fixture:
        monkeypatch.setattr(
            "agrofarm_data.connectors.chirps.rasterio.open",
            lambda _path: fixture,
        )
        with pytest.raises(ChirpsError, match="outside CHIRPS"):
            _read_remote_window("https://chirps.example/day.tif", (-120, 10, -119.8, 10.2))


def test_fetch_uses_official_daily_final_rnl_url(tmp_path: Path) -> None:
    urls: list[str] = []

    def recorded(url: str, bbox: tuple[float, float, float, float]) -> bytes:
        assert bbox == BOX
        urls.append(url)
        return FIXTURE.read_bytes()

    connector = ChirpsConnector(BOX, DAY, DAY, tmp_path, recorded)
    files = connector.fetch()
    assert len(files) == 1
    assert urls == [
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/2024/"
        "chirps-v3.0.rnl.2024.01.01.tif"
    ]
    assert files[0].read_bytes() == FIXTURE.read_bytes()


def test_clean_writes_exact_grid_and_daily_manifest(tmp_path: Path) -> None:
    connector = ChirpsConnector(
        BOX, DAY, DAY, tmp_path / "downloads", lambda _url, _bbox: FIXTURE.read_bytes()
    )
    manifest_path = run_job(connector, tmp_path / "raw", tmp_path / "clean")
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_parquet(tmp_path / "clean" / manifest["file"])
    assert set(frame) == {"date", "latitude", "longitude", "rainfall_mm"}
    assert len(frame) == 16
    assert frame["rainfall_mm"].max() == pytest.approx(0.00167224, abs=1e-7)
    assert manifest["date_range"] == {"start": "2024-01-01", "end": "2024-01-01"}
    assert manifest["spatial_resolution"] == "0.05 degrees, EPSG:4326"
    assert "disaggregated from pentads" in manifest["limitations"][0]
    assert "not valid for one-day extremes" in manifest["limitations"][0]
    assert manifest["source_checksum_sha256"]
    with rasterio.open(tmp_path / "clean" / manifest["layers"][0]["file"]) as raster:
        assert raster.driver == "GTiff"
        assert raster.crs.to_string() == "EPSG:4326"
        assert raster.transform.a == pytest.approx(0.05)
        assert raster.transform.e == pytest.approx(-0.05)
        assert raster.transform.c == pytest.approx(-90.60)
        assert raster.transform.f == pytest.approx(14.60)
        assert raster.profile["tiled"] is True
        assert np.nanmax(raster.read(1)) == pytest.approx(frame["rainfall_mm"].max())


def test_nodata_becomes_null_and_negative_rainfall_is_rejected(tmp_path: Path) -> None:
    def altered(value: float) -> bytes:
        path = tmp_path / "altered.tif"
        with rasterio.open(FIXTURE) as source:
            profile = source.profile
            data = source.read(1)
        data[0, 0] = value
        with rasterio.open(path, "w", **profile) as out:
            out.write(data, 1)
        return path.read_bytes()

    missing = ChirpsConnector(
        BOX, DAY, DAY, tmp_path / "missing", lambda _url, _bbox: altered(-9999)
    )
    assert bool(missing.clean(missing.fetch())["rainfall_mm"].isna().iloc[0])
    bad = ChirpsConnector(BOX, DAY, DAY, tmp_path / "bad", lambda _url, _bbox: altered(-2))
    with pytest.raises(ChirpsError, match="negative rainfall"):
        bad.clean(bad.fetch())


def test_empty_response_and_missing_day_are_rejected(tmp_path: Path) -> None:
    empty = ChirpsConnector(BOX, DAY, DAY, tmp_path / "empty", lambda _url, _bbox: b"")
    with pytest.raises(ChirpsError, match="empty"):
        empty.fetch()
    connector = ChirpsConnector(
        BOX, DAY, date(2024, 1, 2), tmp_path, lambda _url, _bbox: FIXTURE.read_bytes()
    )
    with pytest.raises(ChirpsError, match="missing days"):
        connector.clean([FIXTURE])


def test_invalid_area_and_date_range_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ChirpsError, match="bounding box"):
        ChirpsConnector((-91, 15, -90, 14), DAY, DAY, tmp_path)
    with pytest.raises(ChirpsError, match="date range"):
        ChirpsConnector(BOX, date(2024, 1, 2), DAY, tmp_path)


def test_duplicate_dates_are_rejected(tmp_path: Path) -> None:
    connector = ChirpsConnector(BOX, DAY, DAY, tmp_path, lambda _url, _bbox: FIXTURE.read_bytes())
    with pytest.raises(ChirpsError, match="duplicate dates"):
        connector.clean([FIXTURE, FIXTURE])


def test_requested_area_is_snapped_outward_to_source_cells(tmp_path: Path) -> None:
    inside = (-90.575, 14.425, -90.425, 14.575)
    connector = ChirpsConnector(
        inside, DAY, DAY, tmp_path, lambda _url, _bbox: FIXTURE.read_bytes()
    )
    frame = connector.clean(connector.fetch())
    assert len(frame) == 16
    assert frame["longitude"].min() == pytest.approx(-90.575)
    assert frame["latitude"].max() == pytest.approx(14.575)


def test_broken_geotiff_is_rejected(tmp_path: Path) -> None:
    connector = ChirpsConnector(BOX, DAY, DAY, tmp_path, lambda _url, _bbox: b"not a GeoTIFF")
    with pytest.raises(ChirpsError, match="invalid GeoTIFF"):
        connector.clean(connector.fetch())


def test_command_line_runs_the_job_and_reports_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agrofarm_data.cli import main
    from agrofarm_data.connectors import chirps

    monkeypatch.setattr(chirps, "_read_remote_window", lambda _url, _bbox: FIXTURE.read_bytes())
    period = ["--start", "2024-01-01", "--end", "2024-01-01"]
    folders = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")]
    args = ["chirps", "--bbox", "-90.6", "14.4", "-90.4", "14.6", *period, *folders]
    assert main(args) == 0
    assert (tmp_path / "clean" / "climate_chirps_daily.parquet").is_file()
    assert "16 cells" in capsys.readouterr().out
    assert main(args) == 0  # running it again changes nothing and does not fail


def test_command_line_reports_a_bad_area_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agrofarm_data.cli import main

    period = ["--start", "2024-01-01", "--end", "2024-01-01"]
    folders = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")]
    args = ["chirps", "--bbox", "-90.4", "14.5", "-90.6", "14.7", *period, *folders]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
