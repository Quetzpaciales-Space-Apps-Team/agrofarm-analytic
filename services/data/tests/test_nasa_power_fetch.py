"""Recorded NASA POWER responses exercise point and native regional grids offline."""

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from agrofarm_data.connector import run_job
from agrofarm_data.connectors.nasa_power import (
    PowerError,
    PowerPointConnector,
    PowerRegionalConnector,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nasa_power"


def transport(url: str) -> bytes:
    query = parse_qs(urlparse(url).query)
    if urlparse(url).path.endswith("/point"):
        return (FIXTURES / "point.json").read_bytes()
    parameter = query["parameters"][0]
    return (FIXTURES / f"regional_{parameter}.json").read_bytes()


def test_point_fetch_normalizes_all_parameters_and_writes_manifest(tmp_path: Path) -> None:
    urls: list[str] = []

    def recording_transport(url: str) -> bytes:
        urls.append(url)
        return transport(url)

    connector = PowerPointConnector(
        14.59,
        -90.53,
        date(2024, 1, 1),
        date(2024, 1, 1),
        tmp_path / "downloads",
        recording_transport,
    )
    manifest_path = run_job(connector, tmp_path / "raw", tmp_path / "clean")
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_parquet(tmp_path / "clean" / manifest["file"])
    assert len(urls) == 1
    query = parse_qs(urlparse(urls[0]).query)
    assert query["community"] == ["AG"]
    assert len(query["parameters"][0].split(",")) == 7
    assert frame.loc[0, "temperature_mean_c"] == 19.21
    assert frame.loc[0, "wind_speed_kmh"] == pytest.approx(7.74)
    assert manifest["source_version"] == "POWER Daily API v2.10.3"
    assert manifest["source_checksum_sha256"]


def test_regional_fetch_preserves_each_native_grid(tmp_path: Path) -> None:
    urls: list[str] = []

    def recording_transport(url: str) -> bytes:
        urls.append(url)
        return transport(url)

    connector = PowerRegionalConnector(
        14,
        16,
        -91,
        -89,
        date(2024, 1, 1),
        date(2024, 1, 1),
        tmp_path / "downloads",
        recording_transport,
    )
    manifest_path = run_job(connector, tmp_path / "raw", tmp_path / "clean")
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_parquet(tmp_path / "clean" / manifest["file"])
    assert len(urls) == 7
    assert all(len(parse_qs(urlparse(url).query)["parameters"][0].split(",")) == 1 for url in urls)
    assert all(parse_qs(urlparse(url).query)["community"] == ["AG"] for url in urls)
    assert len(frame[frame["variable"] == "temperature_mean_c"]) == 15
    assert len(frame[frame["variable"] == "solar_radiation_mj_m2"]) == 4
    assert len(frame) == 94
    assert frame.loc[frame["variable"] == "wind_speed_kmh", "value"].max() > 3
    assert len(manifest["source_checksum_sha256"]) == 7


def test_point_fill_value_becomes_null(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "point.json").read_text())
    payload["properties"]["parameter"]["RH2M"]["20240101"] = -999
    connector = PowerPointConnector(
        14.59,
        -90.53,
        date(2024, 1, 1),
        date(2024, 1, 1),
        tmp_path,
        lambda _: json.dumps(payload).encode(),
    )
    frame = connector.clean(connector.fetch())
    assert bool(frame["relative_humidity_pct"].isna().iloc[0])


@pytest.mark.parametrize(
    "change,match", [("empty", "empty"), ("missing", "missing days"), ("extra", "unexpected dates")]
)
def test_point_rejects_bad_daily_series(tmp_path: Path, change: str, match: str) -> None:
    payload = json.loads((FIXTURES / "point.json").read_text())
    if change == "empty":
        payload["properties"]["parameter"] = {}
    elif change == "missing":
        payload["properties"]["parameter"]["T2M"] = {}
    else:
        payload["properties"]["parameter"]["T2M"]["20240102"] = 19.21
    connector = PowerPointConnector(
        14.59,
        -90.53,
        date(2024, 1, 1),
        date(2024, 1, 1),
        tmp_path,
        lambda _: json.dumps(payload).encode(),
    )
    with pytest.raises(PowerError, match=match):
        connector.clean(connector.fetch())


def test_regional_rejects_duplicate_cell(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "regional_T2M.json").read_text())
    payload["features"].append(payload["features"][0])

    def duplicate_transport(url: str) -> bytes:
        if parse_qs(urlparse(url).query)["parameters"] == ["T2M"]:
            return json.dumps(payload).encode()
        return transport(url)

    connector = PowerRegionalConnector(
        14, 16, -91, -89, date(2024, 1, 1), date(2024, 1, 1), tmp_path, duplicate_transport
    )
    with pytest.raises(PowerError, match="duplicate"):
        connector.clean(connector.fetch())


def test_regional_bbox_must_span_two_degrees(tmp_path: Path) -> None:
    with pytest.raises(PowerError, match="2 degrees"):
        PowerRegionalConnector(
            14, 15, -91, -89, date(2024, 1, 1), date(2024, 1, 1), tmp_path, transport
        )
