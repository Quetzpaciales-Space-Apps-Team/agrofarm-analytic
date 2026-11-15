"""Raw store: write once, never modify, never overwrite."""

import stat
from pathlib import Path

import pytest

from agrofarm_data.raw_store import RawModifiedError, store_raw
from tests.fakes import make_source, sha256


def test_file_is_copied_to_dataset_and_version_folder(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    stored = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    assert stored == tmp_path / "raw" / "fake_daily" / "v1" / "fake.csv"
    assert stored.read_bytes() == source.read_bytes()


def test_sha256_is_recorded_in_a_sidecar(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    stored = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    sidecar = stored.with_name("fake.csv.sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == sha256(source)


def test_stored_file_and_sidecar_are_read_only(tmp_path: Path) -> None:
    stored = store_raw(make_source(tmp_path / "in"), tmp_path / "raw", "fake_daily", "v1")
    for path in (stored, stored.with_name("fake.csv.sha256")):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_storing_the_same_content_again_is_a_no_op(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    first = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    mtime = first.stat().st_mtime_ns
    second = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    assert second == first
    assert second.stat().st_mtime_ns == mtime
    assert sorted(p.name for p in first.parent.iterdir()) == ["fake.csv", "fake.csv.sha256"]


def test_same_name_with_different_content_is_refused_and_never_overwrites(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    stored = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    original = stored.read_bytes()
    other = make_source(tmp_path / "other")
    other.write_text("date,value\n2024-01-01,999\n", encoding="utf-8")
    with pytest.raises(RawModifiedError, match=r"fake\.csv"):
        store_raw(other, tmp_path / "raw", "fake_daily", "v1")
    assert stored.read_bytes() == original
    assert sorted(p.name for p in stored.parent.iterdir()) == ["fake.csv", "fake.csv.sha256"]


def test_new_version_is_a_new_folder(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    other = make_source(tmp_path / "other")
    other.write_text("date,value\n2024-01-01,999\n", encoding="utf-8")
    stored = store_raw(other, tmp_path / "raw", "fake_daily", "v2")
    assert stored.parent.name == "v2"


def test_file_edited_after_storing_is_detected(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    stored = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    stored.chmod(0o644)
    stored.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RawModifiedError, match="no longer matches"):
        store_raw(source, tmp_path / "raw", "fake_daily", "v1")


def test_existing_file_without_sidecar_is_checked_by_content(tmp_path: Path) -> None:
    source = make_source(tmp_path / "in")
    stored = store_raw(source, tmp_path / "raw", "fake_daily", "v1")
    sidecar = stored.with_name("fake.csv.sha256")
    sidecar.chmod(0o644)
    sidecar.unlink()
    assert store_raw(source, tmp_path / "raw", "fake_daily", "v1") == stored
    assert sidecar.is_file()


def test_missing_source_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store_raw(tmp_path / "nope.csv", tmp_path / "raw", "fake_daily", "v1")


@pytest.mark.parametrize(
    ("dataset", "version"), [("../x", "v1"), ("a", "a/b"), ("", "v1"), ("a", "..")]
)
def test_dataset_and_version_must_be_plain_folder_names(
    tmp_path: Path, dataset: str, version: str
) -> None:
    with pytest.raises(ValueError, match="plain folder name"):
        store_raw(make_source(tmp_path / "in"), tmp_path / "raw", dataset, version)
