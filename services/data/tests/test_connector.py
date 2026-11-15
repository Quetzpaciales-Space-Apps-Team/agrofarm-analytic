"""Connector interface and `run_job`: the job is idempotent."""

import json
import stat
from pathlib import Path

import pandas as pd
import pytest

from agrofarm_data.connector import Connector, run_job
from agrofarm_data.manifest import ManifestError, validate_manifest
from agrofarm_data.raw_store import RawModifiedError
from tests.fakes import FakeConnector, make_source, sha256


def test_the_fake_connector_satisfies_the_interface(tmp_path: Path) -> None:
    connector: Connector = FakeConnector(make_source(tmp_path / "in"))
    assert connector.dataset == "fake_daily"
    assert connector.version == "v1"


def test_run_job_stores_raw_cleans_and_writes_a_valid_manifest(tmp_path: Path) -> None:
    connector = FakeConnector(make_source(tmp_path / "in"))
    manifest_path = run_job(connector, tmp_path / "raw", tmp_path / "clean")
    manifest = validate_manifest(manifest_path)
    raw = tmp_path / "raw" / "fake_daily" / "v1" / "fake.csv"
    assert stat.S_IMODE(raw.stat().st_mode) == 0o444
    assert manifest["source_checksum_sha256"] == {"fake.csv": sha256(raw)}
    table = pd.read_parquet(manifest_path.parent / manifest["file"])
    assert [d.date().isoformat() for d in table["date"]] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert manifest["checksum_sha256"] == sha256(manifest_path.parent / manifest["file"])


def test_clean_reads_the_stored_raw_copy_not_the_original(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    seen: list[Path] = []

    class Spy(FakeConnector):
        def clean(self, raw_files: list[Path]) -> pd.DataFrame:
            seen.extend(raw_files)
            return super().clean(raw_files)

    run_job(Spy(source), tmp_path / "raw", tmp_path / "clean")
    assert seen == [tmp_path / "raw" / "fake_daily" / "v1" / "fake.csv"]


def test_running_the_job_twice_does_not_duplicate_anything(tmp_path: Path) -> None:
    connector = FakeConnector(make_source(tmp_path / "in"))
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    first = run_job(connector, raw, clean)
    table = clean / "fake_daily.parquet"
    first_checksum, first_manifest = sha256(table), first.read_text(encoding="utf-8")
    second = run_job(connector, raw, clean)

    assert second == first
    assert connector.fetch_calls == 2
    assert sorted(p.name for p in (raw / "fake_daily" / "v1").iterdir()) == [
        "fake.csv",
        "fake.csv.sha256",
    ]
    assert sorted(p.name for p in clean.iterdir()) == [
        "fake_daily.manifest.json",
        "fake_daily.parquet",
    ]
    assert sha256(table) == first_checksum
    assert second.read_text(encoding="utf-8") == first_manifest
    assert json.loads(first_manifest)["checksum_sha256"] == first_checksum


def test_a_changed_source_under_the_same_version_stops_the_job(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    run_job(FakeConnector(source), raw, clean)
    source.write_text("date,value\n2024-01-01,999\n", encoding="utf-8")
    with pytest.raises(RawModifiedError):
        run_job(FakeConnector(source), raw, clean)
    assert "999" not in (raw / "fake_daily" / "v1" / "fake.csv").read_text(encoding="utf-8")


def test_a_connector_that_writes_an_invalid_manifest_fails_the_job(tmp_path: Path) -> None:
    class Bad(FakeConnector):
        def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
            path = super().write_clean(frame, out_dir, raw_files)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            del manifest["license"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return path

    with pytest.raises(ManifestError, match="license"):
        run_job(Bad(make_source(tmp_path / "in")), tmp_path / "raw", tmp_path / "clean")


def test_a_connector_without_raw_files_is_rejected(tmp_path: Path) -> None:
    class Empty(FakeConnector):
        def fetch(self) -> list[Path]:
            return []

    with pytest.raises(ValueError, match="no raw files"):
        run_job(Empty(make_source(tmp_path / "in")), tmp_path / "raw", tmp_path / "clean")
