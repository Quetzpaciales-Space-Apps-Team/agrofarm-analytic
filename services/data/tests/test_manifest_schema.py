"""The manifest JSON Schema and `validate_manifest`."""

import copy
import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from agrofarm_data.manifest import ManifestError, load_schema, validate_manifest
from tests.fakes import valid_manifest


def without(key: str) -> dict[str, Any]:
    manifest = valid_manifest()
    del manifest[key]
    return manifest


def test_schema_is_packaged_data_and_is_itself_valid_json_schema() -> None:
    text = resources.files("agrofarm_data").joinpath("manifest.schema.json").read_text("utf-8")
    assert json.loads(text) == load_schema()


def test_a_complete_manifest_is_accepted_as_dict_or_file(tmp_path: Path) -> None:
    assert validate_manifest(valid_manifest())["dataset"] == "fake_daily"
    path = tmp_path / "m.json"
    path.write_text(json.dumps(valid_manifest()), encoding="utf-8")
    assert validate_manifest(path)["dataset"] == "fake_daily"


@pytest.mark.parametrize(
    "key",
    ["dataset", "source", "license", "spatial_resolution", "checksum_sha256", "processing_version"],
)
def test_required_field_is_reported_by_name(key: str) -> None:
    with pytest.raises(ManifestError, match=key):
        validate_manifest(without(key))


def test_file_or_artifacts_is_required() -> None:
    with pytest.raises(ManifestError, match="file.*artifacts"):
        validate_manifest(without("file"))
    manifest = without("file")
    manifest["artifacts"] = [{"file": "a.parquet", "checksum_sha256": "c" * 64}]
    validate_manifest(manifest)


def test_source_version_or_retrieval_date_is_required() -> None:
    with pytest.raises(ManifestError, match="source_version.*retrieved_at"):
        validate_manifest(without("source_version"))
    manifest = without("source_version")
    manifest["retrieved_at"] = "2025-03-01"
    validate_manifest(manifest)


def test_date_range_or_static_time_step_is_required() -> None:
    manifest = without("date_range")
    with pytest.raises(ManifestError, match="date_range.*time_step"):
        validate_manifest(manifest)
    manifest["time_step"] = "static (2021 classification)"
    validate_manifest(manifest)


def test_daily_time_step_without_date_range_is_not_static() -> None:
    with pytest.raises(ManifestError):
        validate_manifest(without("date_range"))


def test_cc_by_license_requires_attribution() -> None:
    with pytest.raises(ManifestError, match="attribution"):
        validate_manifest(without("attribution"))


def test_other_licenses_do_not_require_a_separate_attribution() -> None:
    manifest = without("attribution")
    manifest["license"] = "Public domain (US Government work)"
    validate_manifest(manifest)


def test_source_checksums_are_required_when_there_is_a_raw_input() -> None:
    with pytest.raises(ManifestError, match="source_checksum_sha256"):
        validate_manifest(without("source_checksum_sha256"))
    manifest = without("source_checksum_sha256")
    manifest["no_raw_input"] = True
    validate_manifest(manifest)


def test_source_checksum_may_be_one_string() -> None:
    manifest = valid_manifest() | {"source_checksum_sha256": "d" * 64}
    validate_manifest(manifest)


@pytest.mark.parametrize("bad", ["abc", "G" * 64, "A" * 64, 5])
def test_checksum_must_be_64_lowercase_hex_characters(bad: object) -> None:
    with pytest.raises(ManifestError, match="checksum_sha256"):
        validate_manifest(valid_manifest() | {"checksum_sha256": bad})


def test_bad_source_checksum_names_the_file() -> None:
    manifest = valid_manifest() | {"source_checksum_sha256": {"fake.csv": "xyz"}}
    with pytest.raises(ManifestError, match=r"source_checksum_sha256\.fake\.csv"):
        validate_manifest(manifest)


def test_dates_must_be_iso() -> None:
    manifest = copy.deepcopy(valid_manifest())
    manifest["date_range"]["start"] = "01/02/2024"
    with pytest.raises(ManifestError, match=r"date_range\.start"):
        validate_manifest(manifest)


def test_empty_strings_are_not_accepted() -> None:
    with pytest.raises(ManifestError, match="processing_version"):
        validate_manifest(valid_manifest() | {"processing_version": ""})


def test_errors_read_as_path_and_message_one_per_line() -> None:
    manifest = valid_manifest() | {"checksum_sha256": "bad", "dataset": 3}
    del manifest["source"]
    with pytest.raises(ManifestError) as info:
        validate_manifest(manifest)
    lines = str(info.value).splitlines()
    assert "checksum_sha256: 'bad' does not match" in " ".join(lines)
    assert any(line.startswith("dataset: 3 is not of type 'string'") for line in lines)
    assert any(line.startswith("(manifest): 'source' is a required property") for line in lines)
    assert info.value.errors == lines


def test_unreadable_files_are_reported_as_manifest_errors(tmp_path: Path) -> None:
    broken = tmp_path / "m.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        validate_manifest(broken)
    with pytest.raises(ManifestError, match="cannot be read"):
        validate_manifest(tmp_path / "missing.json")
    broken.write_text("[]", encoding="utf-8")
    with pytest.raises(ManifestError, match="must be a JSON object"):
        validate_manifest(broken)
