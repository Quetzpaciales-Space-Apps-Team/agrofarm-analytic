"""Meteomatics point forecast: every response cached, a daily budget, credentials never stored."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from agrofarm_data.cli import main
from agrofarm_data.connectors.meteomatics import (
    COLUMNS,
    BudgetExceededError,
    MeteomaticsClient,
    MeteomaticsError,
    write_clean,
)
from agrofarm_data.manifest import validate_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "meteomatics" / "forecast_7_days.json"
DAY = dt.date(2026, 11, 15)
LAT, LON = 14.59, -90.53


class FakeHttp:
    """Stands in for the API: counts calls and keeps the last URL and credentials."""

    def __init__(self, body: str | None = None, error: Exception | None = None) -> None:
        self.body = FIXTURE.read_text("utf-8") if body is None else body
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, url: str, username: str, password: str) -> str:
        self.calls.append((url, username, password))
        if self.error:
            raise self.error
        return self.body


def client(tmp_path: Path, http: FakeHttp, limit: int = 5, day: dt.date = DAY) -> MeteomaticsClient:
    return MeteomaticsClient(
        cache_dir=tmp_path / "cache", username="team-user", password="team-secret",  # noqa: S106
        daily_limit=limit, http=http, today=lambda: day,
    )  # fmt: skip


def test_forecast_returns_one_row_per_day_in_team_units(tmp_path: Path) -> None:
    frame = client(tmp_path, FakeHttp()).forecast(LAT, LON, days=7)
    assert list(frame.columns) == COLUMNS
    assert [d.date().isoformat() for d in frame["date"]][:2] == ["2026-11-15", "2026-11-16"]
    assert len(frame) == 7
    first = frame.iloc[0]
    assert first["temperature_mean_c"] == 20.2
    assert first["temperature_min_c"] == 15.3
    assert first["temperature_max_c"] == 27.1
    assert first["rainfall_mm"] == 0.1
    assert first["wind_speed_kmh"] == 7.2
    assert first["relative_humidity_pct"] == 78.3
    assert first["solar_radiation_mj_m2"] == pytest.approx(20.719, abs=0.001)  # J/m2 to MJ/m2
    assert set(frame["latitude"]) == {LAT}


def test_repeated_call_is_served_from_the_cache_without_a_request(tmp_path: Path) -> None:
    http = FakeHttp()
    api = client(tmp_path, http)
    first = api.forecast(LAT, LON, days=7)
    second = api.forecast(LAT, LON, days=7)
    again = client(tmp_path, http).forecast(LAT, LON, days=7)  # a new process, same cache
    assert len(http.calls) == 1
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, again)
    assert api.requests_today() == 1


def test_a_different_point_or_day_is_a_new_request(tmp_path: Path) -> None:
    http = FakeHttp()
    client(tmp_path, http).forecast(LAT, LON, days=7)
    client(tmp_path, http).forecast(15.0, -91.0, days=7)
    client(tmp_path, http, day=DAY + dt.timedelta(days=1)).forecast(LAT, LON, days=7)
    assert len(http.calls) == 3


def test_budget_refuses_calls_beyond_the_daily_limit(tmp_path: Path) -> None:
    http = FakeHttp()
    api = client(tmp_path, http, limit=2)
    api.forecast(14.0, -90.0)
    api.forecast(14.1, -90.0)
    with pytest.raises(BudgetExceededError, match="2"):
        api.forecast(14.2, -90.0)
    assert len(http.calls) == 2
    api.forecast(14.0, -90.0)  # cached answers stay available when the budget is spent


def test_budget_starts_again_the_next_day(tmp_path: Path) -> None:
    http = FakeHttp()
    client(tmp_path, http, limit=1).forecast(14.0, -90.0)
    client(tmp_path, http, limit=1, day=DAY + dt.timedelta(days=1)).forecast(14.0, -90.0)
    assert len(http.calls) == 2


def test_failed_requests_count_against_the_budget_and_are_not_cached(tmp_path: Path) -> None:
    broken = FakeHttp(error=OSError("HTTP 500"))
    api = client(tmp_path, broken, limit=3)
    with pytest.raises(MeteomaticsError):
        api.forecast(LAT, LON)
    assert api.requests_today() == 1
    assert len(list((tmp_path / "cache").glob("responses/*.json"))) == 0


def test_credentials_are_sent_but_never_written_or_put_in_the_url(tmp_path: Path) -> None:
    http = FakeHttp()
    client(tmp_path, http).forecast(LAT, LON)
    url, username, password = http.calls[0]
    assert (username, password) == ("team-user", "team-secret")
    assert "team-user" not in url and "team-secret" not in url
    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            text = path.read_text("utf-8")
            assert "team-secret" not in text and "team-user" not in text


def test_user_name_in_the_response_is_not_cached(tmp_path: Path) -> None:
    body = json.loads(FIXTURE.read_text("utf-8"))
    body["user"] = "account-name"
    client(tmp_path, FakeHttp(json.dumps(body))).forecast(LAT, LON)
    stored = "".join(p.read_text("utf-8") for p in (tmp_path / "cache").rglob("*.json"))
    assert "account-name" not in stored


def test_error_message_does_not_leak_credentials(tmp_path: Path) -> None:
    broken = FakeHttp(error=OSError("401 for team-user:team-secret"))
    with pytest.raises(MeteomaticsError) as raised:
        client(tmp_path, broken).forecast(LAT, LON)
    assert "team-secret" not in str(raised.value) and "team-user" not in str(raised.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not json", "not JSON"),
        (json.dumps({"status": "ERROR", "message": "bad parameter"}), "bad parameter"),
        (json.dumps({"status": "OK", "data": []}), "no data"),
    ],
    ids=["not_json", "api_error", "empty"],
)
def test_bad_responses_are_rejected_and_not_cached(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(MeteomaticsError, match=message):
        client(tmp_path, FakeHttp(body)).forecast(LAT, LON)
    assert len(list((tmp_path / "cache").glob("responses/*.json"))) == 0


def test_missing_marker_becomes_null(tmp_path: Path) -> None:
    body = json.loads(FIXTURE.read_text("utf-8"))
    body["data"][3]["coordinates"][0]["dates"][0]["value"] = -999
    frame = client(tmp_path, FakeHttp(json.dumps(body))).forecast(LAT, LON)
    assert bool(frame["rainfall_mm"].isna().iloc[0])


@pytest.mark.parametrize(("lat", "lon", "days"), [(95.0, 0.0, 7), (0.0, 200.0, 7), (0.0, 0.0, 0),
                                                  (0.0, 0.0, 20)])  # fmt: skip
def test_invalid_arguments_are_rejected_before_any_request(
    tmp_path: Path, lat: float, lon: float, days: int
) -> None:
    http = FakeHttp()
    with pytest.raises(ValueError):
        client(tmp_path, http).forecast(lat, lon, days=days)
    assert http.calls == []


def test_missing_credentials_are_reported_without_a_request(tmp_path: Path) -> None:
    http = FakeHttp()
    api = MeteomaticsClient(cache_dir=tmp_path, username="", password="", http=http)
    with pytest.raises(MeteomaticsError, match="METEOMATICS_USERNAME"):
        api.forecast(LAT, LON)
    assert http.calls == []


def test_clean_output_has_a_valid_manifest_that_says_it_is_a_forecast(tmp_path: Path) -> None:
    api = client(tmp_path, FakeHttp())
    frame = api.forecast(LAT, LON, days=7)
    manifest_path = write_clean(frame, tmp_path / "clean", api.last_response_file)
    validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["rows"] == 7
    assert manifest["retrieved_at"] == "2026-11-15"
    assert manifest["date_range"] == {"start": "2026-11-15", "end": "2026-11-21"}
    assert any("forecast" in text and "not an observation" in text for text in manifest["limitations"])
    assert "team-secret" not in manifest_path.read_text("utf-8")


def test_command_line_reads_credentials_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("METEOMATICS_USERNAME", raising=False)
    monkeypatch.delenv("METEOMATICS_PASSWORD", raising=False)
    args = ["meteomatics-forecast", "--lat", "14.59", "--lon", "-90.53", "--cache",
            str(tmp_path / "cache"), "--out", str(tmp_path / "clean")]  # fmt: skip
    assert main(args) == 2
    assert "METEOMATICS_USERNAME" in capsys.readouterr().err
