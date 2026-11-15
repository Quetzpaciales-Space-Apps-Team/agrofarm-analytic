"""Only the pinned, owner-accepted, intact NDVI bundle reaches deployment."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_ndvi", Path(__file__).resolve().parents[1] / "deploy/prepare_ndvi.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("failure", [None, "archive", "member", "acceptance", "weights"])
def test_verified_release_only(tmp_path: Path, failure: str | None) -> None:
    data = {name: b"test" for name in module.FILES - {"serving_card.json"}}
    card: dict[str, Any] = {
        "model_version": "test",
        "owner_accepted_limited_use": failure != "acceptance",
        "files_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()},
    }
    data["serving_card.json"] = json.dumps(card).encode()
    if failure == "weights":
        data["p50.lgb.txt"] = b"changed"
    if failure == "member":
        data["../outside"] = b"bad"
    source = tmp_path / "model.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, raw in data.items():
            info = tarfile.TarInfo("ndvi/0.1/" + name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    manifest = {
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "directory": "ndvi/0.1",
        "model_version": "test",
    }
    if failure == "archive":
        manifest["archive_sha256"] = "wrong"
    if failure:
        with pytest.raises(ValueError):
            module.prepare(source, tmp_path / "out", manifest)
        assert not (tmp_path / "out").exists()
    else:
        module.prepare(source, tmp_path / "out", manifest)
        assert (tmp_path / "out/ndvi/0.1/p50.lgb.txt").read_bytes() == b"test"
