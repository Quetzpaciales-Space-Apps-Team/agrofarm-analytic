"""DS-020: the `nasa-power --fetch` command, the polite User-Agent and the regional limits.

Offline: the transport is replaced by the recorded fixtures. The live API was checked once on
2026-11-15 (see tests/fixtures/nasa_power/README.md).
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from agrofarm_data import cli
from agrofarm_data.connectors import nasa_power
from agrofarm_data.connectors.nasa_power import PowerError, PowerRegionalConnector
from agrofarm_data.manifest import validate_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "nasa_power"
DAY = ["--start", "2024-01-01", "--end", "2024-01-01"]


def transport(url: str) -> bytes:
    if urlparse(url).path.endswith("/point"):
        return (FIXTURES / "point.json").read_bytes()
    return (
        FIXTURES / f"regional_{parse_qs(urlparse(url).query)['parameters'][0]}.json"
    ).read_bytes()


def test_fetch_point_writes_a_valid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["nasa-power", "--fetch", "point", "--lat", "14.59", "--lon", "-90.53", *DAY,
         "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")],
        power_transport=transport,
    )  # fmt: skip
    assert code == 0
    manifest_path = tmp_path / "clean" / "climate_nasa_power_daily.manifest.json"
    validate_manifest(manifest_path)
    assert "1 rows" in capsys.readouterr().out
    assert len(pd.read_parquet(tmp_path / "clean" / "climate_nasa_power_daily.parquet")) == 1


def test_fetch_regional_writes_a_valid_manifest_with_the_area_limits(tmp_path: Path) -> None:
    code = cli.main(
        ["nasa-power", "--fetch", "regional", "--bbox", "-91", "14", "-89", "16", *DAY,
         "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")],
        power_transport=transport,
    )  # fmt: skip
    assert code == 0
    path = tmp_path / "clean" / "climate_nasa_power_daily_regional.manifest.json"
    validate_manifest(path)
    manifest: dict[str, Any] = json.loads(path.read_text())
    text = " ".join(manifest["limitations"])
    assert "2 to 10 degrees" in text
    assert "one request per variable" in text


def test_fetch_point_manifest_lists_limitations_too(tmp_path: Path) -> None:
    cli.main(
        ["nasa-power", "--fetch", "point", "--lat", "14.59", "--lon", "-90.53", *DAY,
         "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")],
        power_transport=transport,
    )  # fmt: skip
    path = tmp_path / "clean" / "climate_nasa_power_daily.manifest.json"
    assert "reanalysis" in " ".join(json.loads(path.read_text())["limitations"])


def test_fetch_needs_its_arguments(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    base = ["nasa-power", "--raw", str(tmp_path), "--out", str(tmp_path)]
    for argv in (
        [*base, "--fetch", "point", *DAY],  # no coordinates
        [*base, "--fetch", "regional", *DAY],  # no bbox
        [*base, "--fetch", "point", "--lat", "1", "--lon", "1"],  # no dates
        [*base, "--input", "x.csv", "--fetch", "point"],  # both modes
        [*base],  # neither mode
    ):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv, power_transport=transport)
        assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_fetch_reports_a_power_error_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["nasa-power", "--fetch", "point", "--lat", "14.59", "--lon", "-90.53", *DAY,
         "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "clean")],
        power_transport=lambda url: b'{"header": {}}',
    )  # fmt: skip
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


def test_download_sends_the_connector_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        seen["url"] = request.full_url
        return Response()

    monkeypatch.setattr(nasa_power, "urlopen", fake_urlopen)
    assert nasa_power._download("https://power.larc.nasa.gov/api/x") == b"{}"
    assert seen["agent"] == "agrofarm-data-connector/0.1"


def test_regional_box_is_limited_to_ten_degrees(tmp_path: Path) -> None:
    from datetime import date

    with pytest.raises(PowerError, match="10 degrees"):
        PowerRegionalConnector(10, 22, -91, -89, date(2024, 1, 1), date(2024, 1, 1), tmp_path)
    with pytest.raises(PowerError, match="10 degrees"):
        PowerRegionalConnector(10, 14, -96, -84, date(2024, 1, 1), date(2024, 1, 1), tmp_path)
    PowerRegionalConnector(10, 20, -96, -86, date(2024, 1, 1), date(2024, 1, 1), tmp_path)
