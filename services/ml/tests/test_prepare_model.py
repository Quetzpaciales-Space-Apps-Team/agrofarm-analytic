"""Release bytes are verified before any model enters a deployment image."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_model", Path(__file__).resolve().parents[1] / "deploy/prepare_model.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def bundle(path: Path, *, extra: str | None = None) -> dict[str, str]:
    weights = b"test weights"
    card = {
        "model_version": "maize-leaf-lightgbm-0.1.1",
        "feature_version": "rgb-hsv-texture-2",
        "internal_gate_passed": True,
        "model_sha256": hashlib.sha256(weights).hexdigest(),
    }
    contents = {
        "maize-leaf/0.1.1/model.lgb.txt": weights,
        "maize-leaf/0.1.1/model_card.json": json.dumps(card).encode(),
        "maize-leaf/0.1.1/split_manifest.json": b"[]",
    }
    if extra:
        contents[extra] = b"unexpected"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_version": str(card["model_version"]),
        "feature_version": str(card["feature_version"]),
        "directory": "maize-leaf/0.1.1",
    }


def test_release_is_verified_then_extracted(tmp_path: Path) -> None:
    source = tmp_path / "model.tar.gz"
    manifest = bundle(source)
    module.prepare(source, tmp_path / "out", manifest)
    assert (tmp_path / "out/maize-leaf/0.1.1/model.lgb.txt").read_bytes() == b"test weights"


def test_bad_checksum_is_rejected_before_extraction(tmp_path: Path) -> None:
    source = tmp_path / "model.tar.gz"
    manifest = bundle(source)
    manifest["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        module.prepare(source, tmp_path / "out", manifest)
    assert not (tmp_path / "out").exists()


def test_unexpected_or_escaping_member_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "model.tar.gz"
    manifest = bundle(source, extra="../outside")
    with pytest.raises(ValueError, match="members"):
        module.prepare(source, tmp_path / "out", manifest)
    assert not (tmp_path / "outside").exists()
