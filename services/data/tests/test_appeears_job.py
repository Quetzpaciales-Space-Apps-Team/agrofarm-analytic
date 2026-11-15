"""AppEEARS connector, the job: bundle files, cleaning, manifest, redirects and the command.

Everything reads the answers recorded from the real API (`tests/fixtures/appeears/`)."""

import hashlib
import io
import json
import shutil
import urllib.request
from http.client import HTTPMessage
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agrofarm_data.cli import main
from agrofarm_data.connector import run_job
from agrofarm_data.connectors.appeears import (
    SPECS,
    AppEearsConnector,
    AppEearsError,
    SourceSpec,
    StripAuthorizationOnRedirect,
    TaskPending,
    select_bundle_files,
)
from agrofarm_data.manifest import validate_manifest
from tests.test_appeears import BOX, FakeTransport, catalog, fixture

FIXTURES = Path(__file__).parent / "fixtures" / "appeears"
TASK_ID = fixture("task_submit_response.json")["task_id"]
LISTING = fixture("bundle_listing.json")
START, END = "2025-06-01", "2025-06-30"


class BundleTransport(FakeTransport):
    """The catalog, the task and the recorded bundle. `download` copies the recorded files."""

    def __init__(self, status: str = "done", tamper: str | None = None) -> None:
        routes = dict(catalog().routes)
        routes[("POST", "/login")] = {"token": "session-token", "token_type": "Bearer"}
        routes[("POST", "/task")] = {"task_id": TASK_ID, "status": "pending"}
        routes[("GET", f"/task/{TASK_ID}")] = {"status": status}
        routes[("GET", f"/bundle/{TASK_ID}")] = LISTING
        super().__init__(routes)
        self.tamper = tamper
        self.downloaded: list[str] = []

    def download(self, path: str, token: str, destination: Path) -> None:
        file_id = path.rsplit("/", 1)[1]
        name = next(f["file_name"] for f in LISTING["files"] if f["file_id"] == file_id)
        self.downloaded.append(name)
        shutil.copy(FIXTURES / name, destination)
        if name == self.tamper:
            destination.write_bytes(destination.read_bytes() + b"\n#tampered")


def connector(
    key: str, tmp_path: Path, transport: BundleTransport | None = None, **extra: Any
) -> AppEearsConnector:
    return AppEearsConnector(
        SPECS[key],
        BOX,
        START,
        END,
        transport=transport or BundleTransport(),
        token="session-token",  # noqa: S106 -- a test value
        state_dir=tmp_path / "state",
        area_name="agrofarm-test",
        retrieved_at="2026-11-15",
        **extra,
    )


def is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def spec_files(spec: SourceSpec) -> set[str]:
    return {f["file_name"] for f in select_bundle_files(LISTING, spec)}


# --- which files of the bundle are used -------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "modis-13q1",
            {
                "MOD13Q1-061-Statistics.csv",
                "MOD13Q1-061-250m-16-days-pixel-reliability-Statistics-QA.csv",
                "MOD13Q1-061-250m-16-days-pixel-reliability-lookup.csv",
            },
        ),
        (
            "viirs-vnp13a1",
            {
                "VNP13A1-002-Statistics.csv",
                "VNP13A1-002-500-m-16-days-pixel-reliability-Statistics-QA.csv",
                "VNP13A1-002-500-m-16-days-pixel-reliability-lookup.csv",
            },
        ),
        ("smap-l4", {"SPL4SMGP-008-Statistics.csv"}),
    ],
)
def test_only_the_statistics_files_of_the_source_are_selected(key: str, expected: set[str]) -> None:
    assert spec_files(SPECS[key]) == expected


def test_the_request_echo_and_the_rasters_are_never_selected() -> None:
    chosen = {f["file_name"] for key in SPECS for f in select_bundle_files(LISTING, SPECS[key])}
    assert not any(name.endswith(("-request.json", ".tif", "README.md")) for name in chosen)


def test_a_missing_statistics_file_is_named() -> None:
    cut = {**LISTING, "files": [f for f in LISTING["files"] if "SPL4SMGP" not in f["file_name"]]}
    with pytest.raises(AppEearsError, match="SPL4SMGP-008-Statistics.csv"):
        select_bundle_files(cut, SPECS["smap-l4"])


# --- fetch --------------------------------------------------------------------------------------


def test_fetch_downloads_the_selected_files_and_a_listing_without_signed_urls(
    tmp_path: Path,
) -> None:
    transport = BundleTransport()
    files = connector("modis-13q1", tmp_path, transport).fetch()
    names = {p.name for p in files}
    assert names == spec_files(SPECS["modis-13q1"]) | {"bundle_listing.json"}
    assert not any(n.endswith("-request.json") for n in transport.downloaded)
    listing = json.loads(next(p for p in files if p.name == "bundle_listing.json").read_text())
    assert listing["task_id"] == TASK_ID
    assert "s3_url" not in json.dumps(listing)


def test_fetch_never_sends_the_token_anywhere_but_the_api_calls(tmp_path: Path) -> None:
    transport = BundleTransport()
    connector("smap-l4", tmp_path, transport).fetch()
    json_calls = [c for c in transport.calls if c[3] is not None]
    assert all(c[1].startswith(("/task", "/bundle")) for c in json_calls)
    assert not any("session-token" in json.dumps(c[2]) for c in transport.calls)


def test_fetch_stays_pending_without_downloading_when_the_task_is_not_done(tmp_path: Path) -> None:
    transport = BundleTransport(status="processing")
    with pytest.raises(TaskPending):
        connector("smap-l4", tmp_path, transport).fetch()
    assert transport.downloaded == []


def test_fetch_waits_only_when_asked_and_then_gives_up_without_blocking_forever(
    tmp_path: Path,
) -> None:
    transport = BundleTransport(status="processing")
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    with pytest.raises(TaskPending):
        connector(
            "smap-l4",
            tmp_path,
            transport,
            wait_seconds=50,
            poll_seconds=20,
            sleep=sleep,
            clock=lambda: now[0],
        ).fetch()
    assert sleeps == [20, 20, 20]
    assert [c[:2] for c in transport.calls].count(("POST", "/task")) == 1


def test_a_file_that_does_not_match_its_listed_checksum_is_refused(tmp_path: Path) -> None:
    transport = BundleTransport(tamper="MOD13Q1-061-Statistics.csv")
    with pytest.raises(AppEearsError, match=r"MOD13Q1-061-Statistics\.csv.*checksum"):
        connector("modis-13q1", tmp_path, transport).fetch()


def test_the_layers_are_confirmed_against_the_catalog_on_every_fetch(tmp_path: Path) -> None:
    transport = BundleTransport()
    transport.routes[("GET", "/product")] = []
    with pytest.raises(AppEearsError, match=r"MOD13Q1\.061"):
        connector("modis-13q1", tmp_path, transport).fetch()
    assert not [c for c in transport.calls if c[0] == "POST"], "nothing is submitted for a bad id"


# --- cleaning -----------------------------------------------------------------------------------


def cleaned(key: str, tmp_path: Path, **extra: Any) -> pd.DataFrame:
    c = connector(key, tmp_path, **extra)
    return c.clean(c.fetch())


def test_modis_rows_carry_the_statistics_and_flag_a_mostly_cloudy_composite(tmp_path: Path) -> None:
    table = cleaned("modis-13q1", tmp_path).set_index("date")
    assert [d.strftime("%Y-%m-%d") for d in table.index] == [
        "2025-05-25",
        "2025-06-10",
        "2025-06-26",
    ]
    cloudy = table.loc["2025-05-25"]  # 8,213 of 17,280 pixels are good or marginal
    assert cloudy["status"] == "too_cloudy"
    assert cloudy["usable_fraction"] == pytest.approx(8213 / 17280)
    assert pd.isna(cloudy["ndvi_mean"]) and pd.isna(cloudy["evi_mean"])
    clear = table.loc["2025-06-10"]
    assert clear["status"] == "ok"
    assert clear["ndvi_mean"] == pytest.approx(0.538481)
    assert clear["ndvi_median"] == pytest.approx(0.56095)
    assert clear["pixels_total"] == 17280 and clear["pixels_usable"] == 1986 + 8595


def test_viirs_counts_excellent_good_and_acceptable_as_usable_and_reads_blank_counts(
    tmp_path: Path,
) -> None:
    table = cleaned("viirs-vnp13a1", tmp_path).set_index("date")
    qa = pd.read_csv(FIXTURES / "VNP13A1-002-500-m-16-days-pixel-reliability-Statistics-QA.csv")
    first = qa.iloc[0]
    usable = sum(float(first[c]) for c in ("0", "1", "2"))
    assert table.iloc[0]["pixels_usable"] == usable
    # 2025-06-02 has blank counts for the best classes: a blank is zero pixels, not an error.
    assert table.loc["2025-06-02", "pixels_usable"] == 63
    assert table.loc["2025-06-02", "status"] == "too_cloudy"
    assert (table["pixels_total"] == 4320).all()


def test_a_cloudy_composite_never_reports_zero_it_reports_no_value(tmp_path: Path) -> None:
    table = cleaned("viirs-vnp13a1", tmp_path)
    cloudy = table[table["status"] != "ok"]
    assert len(cloudy) > 0
    assert cloudy[["ndvi_mean", "ndvi_median", "ndvi_std", "evi_mean"]].isna().all().all()
    assert not (table["ndvi_mean"] == 0).any()


def test_smap_rows_are_daily_count_weighted_means_of_the_three_hourly_samples(
    tmp_path: Path,
) -> None:
    table = cleaned("smap-l4", tmp_path).set_index("date")
    raw = pd.read_csv(FIXTURES / "SPL4SMGP-008-Statistics.csv")
    raw["day"] = raw["Date"].str[:10]
    assert len(table) == 30 and table.index.is_monotonic_increasing
    first = raw[(raw["day"] == "2025-06-01") & (raw["Dataset"] == "Geophysical_Data_sm_surface")]
    assert len(first) == 8
    assert table.loc["2025-06-01", "sm_surface_mean"] == pytest.approx(first["Mean"].mean())
    assert table.loc["2025-06-01", "samples_surface"] == 8
    assert table.loc["2025-06-01", "status"] == "ok"
    assert table["sm_surface_mean"].between(0, 0.9).all()
    assert table["sm_rootzone_mean"].between(0, 0.9).all()


def test_a_day_with_too_few_samples_has_no_value(tmp_path: Path) -> None:
    c = connector("smap-l4", tmp_path)
    files = c.fetch()
    path = next(p for p in files if p.name == "SPL4SMGP-008-Statistics.csv")
    table = pd.read_csv(path)
    drop = table[
        (table["Date"].str[:10] == "2025-06-03") & (table["Dataset"].str.endswith("surface"))
    ]
    table.drop(drop.index[:4]).to_csv(path, index=False)  # leaves 4 of 8 samples
    result = c.clean(files).set_index("date")
    assert result.loc["2025-06-03", "status"] == "too_few_samples"
    assert is_missing(result.loc["2025-06-03", "sm_surface_mean"])
    assert result.loc["2025-06-03", "samples_surface"] == 4
    assert result.loc["2025-06-02", "status"] == "ok"


def test_a_sample_outside_the_valid_range_or_with_no_pixels_is_dropped(tmp_path: Path) -> None:
    c = connector("smap-l4", tmp_path)
    files = c.fetch()
    path = next(p for p in files if p.name == "SPL4SMGP-008-Statistics.csv")
    table = pd.read_csv(path)
    day = (table["Date"].str[:10] == "2025-06-05") & (table["Dataset"].str.endswith("rootzone"))
    index = list(table[day].index)
    table.loc[index[0], "Mean"] = -9999.0  # the catalog fill value
    table.loc[index[1], "Count"] = 0.0
    table.to_csv(path, index=False)
    row = c.clean(files).set_index("date").loc["2025-06-05"]
    assert row["samples_rootzone"] == 6
    assert row["sm_rootzone_mean"] >= 0


def test_an_empty_statistics_file_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    c = connector("modis-13q1", tmp_path)
    files = c.fetch()
    path = next(p for p in files if p.name == "MOD13Q1-061-Statistics.csv")
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(AppEearsError, match="no rows"):
        c.clean(files)


# --- manifest and job ------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(SPECS))
def test_the_job_writes_a_manifest_that_follows_the_schema(key: str, tmp_path: Path) -> None:
    c = connector(key, tmp_path)
    manifest_path = run_job(c, tmp_path / "raw", tmp_path / "clean")
    manifest = validate_manifest(manifest_path)
    assert manifest["dataset"] == SPECS[key].dataset
    assert manifest["product"] == SPECS[key].product
    assert manifest["task_id"] == TASK_ID
    assert manifest["bbox"] == list(BOX)
    assert set(manifest["layers"]) == {layer.name for layer in SPECS[key].layers}
    assert manifest["limitations"]
    table = pd.read_parquet(tmp_path / "clean" / f"{SPECS[key].dataset}.parquet")
    assert manifest["rows"] == len(table)
    assert manifest["date_range"]["start"] <= manifest["date_range"]["end"]


def test_the_manifest_records_what_the_catalog_said_and_the_quality_rule(tmp_path: Path) -> None:
    manifest = json.loads(
        run_job(connector("modis-13q1", tmp_path), tmp_path / "raw", tmp_path / "clean").read_text()
    )
    ndvi = manifest["layers"]["_250m_16_days_NDVI"]
    assert ndvi["fill_value"] == -3000 and ndvi["scale_factor"] == pytest.approx(0.0001)
    assert manifest["quality_rule"]["usable_classes"] == [0, 1]
    assert manifest["quality_rule"]["min_usable_fraction"] == 0.5
    assert "the mean includes cloudy pixels" in " ".join(manifest["limitations"]).lower()


def test_nothing_secret_or_personal_reaches_any_output(tmp_path: Path) -> None:
    run_job(connector("smap-l4", tmp_path), tmp_path / "raw", tmp_path / "clean")
    everything = ""
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".txt"}:
            everything += path.read_text(errors="ignore")
    assert "session-token" not in everything
    assert "@" not in everything


def test_running_the_job_again_changes_nothing(tmp_path: Path) -> None:
    first = run_job(connector("viirs-vnp13a1", tmp_path), tmp_path / "raw", tmp_path / "clean")
    before = hashlib.sha256(first.read_bytes()).hexdigest()
    second = run_job(connector("viirs-vnp13a1", tmp_path), tmp_path / "raw", tmp_path / "clean")
    assert hashlib.sha256(second.read_bytes()).hexdigest() == before


# --- redirects ----------------------------------------------------------------------------------


def _redirect(handler: StripAuthorizationOnRedirect, to: str) -> urllib.request.Request | None:
    request = urllib.request.Request(
        "https://appeears.earthdatacloud.nasa.gov/api/bundle/x/y",
        headers={"Authorization": "Bearer session-token", "User-Agent": "test"},
    )
    return handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), to)


def test_the_token_is_not_forwarded_to_another_host_on_a_redirect() -> None:
    other = _redirect(StripAuthorizationOnRedirect(), "https://d2v96idf0qj40l.cloudfront.net/file")
    assert other is not None
    assert "Authorization" not in other.headers and "authorization" not in {
        k.lower() for k in other.headers
    }


def test_the_token_stays_on_a_redirect_inside_the_api_host() -> None:
    same = _redirect(
        StripAuthorizationOnRedirect(), "https://appeears.earthdatacloud.nasa.gov/api/other"
    )
    assert same is not None and same.get_header("Authorization") == "Bearer session-token"


# --- the command --------------------------------------------------------------------------------

ARGS = [
    "appeears", "--source", "modis-13q1", "--bbox", "-90.65", "14.45", "-90.4", "14.75",
    "--start", START, "--end", END,
]  # fmt: skip


def command(tmp_path: Path, extra: list[str] | None = None, **kwargs: Any) -> int:
    paths = ["--raw-dir", str(tmp_path / "raw"), "--out", str(tmp_path / "clean"),
             "--state-dir", str(tmp_path / "state")]  # fmt: skip
    return main([*ARGS, *paths, *(extra or [])], **kwargs)


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EARTHDATA_USERNAME", "someone")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p4ssw0rd-never-printed")


def test_the_command_writes_the_dataset_when_the_task_is_done(
    tmp_path: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert command(tmp_path, appeears_transport=BundleTransport()) == 0
    out = capsys.readouterr()
    assert (tmp_path / "clean" / "appeears_modis_13q1_vegetation.manifest.json").is_file()
    assert "p4ssw0rd-never-printed" not in out.out + out.err


def test_the_command_exits_75_without_blocking_while_the_task_runs(
    tmp_path: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert command(tmp_path, appeears_transport=BundleTransport(status="processing")) == 75
    err = capsys.readouterr().err
    assert TASK_ID in err and "run the same command again" in err
    assert not (tmp_path / "clean").exists()


def test_the_command_names_a_missing_credential_and_never_prints_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EARTHDATA_USERNAME", "someone-private")
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    assert command(tmp_path, appeears_transport=BundleTransport()) == 2
    err = capsys.readouterr().err
    assert "EARTHDATA_PASSWORD" in err and "someone-private" not in err


def test_the_command_rejects_a_bad_box_before_any_call(tmp_path: Path, credentials: None) -> None:
    transport = BundleTransport()
    bad = [a if a != "-90.65" else "-90.1" for a in ARGS]
    with pytest.raises(SystemExit):
        main(
            [*bad, "--raw-dir", str(tmp_path), "--out", str(tmp_path)], appeears_transport=transport
        )
    assert transport.calls == []
