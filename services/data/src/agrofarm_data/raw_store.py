"""Write-once raw store: `raw/<dataset>/<version>/<name>` plus a `.sha256` sidecar.

A raw file is copied once, made read-only and never changed. Storing the same content again
does nothing. The same name with different content raises `RawModifiedError`: a new source
release needs a new version folder, not an overwrite.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

READ_ONLY = 0o444
SIDECAR_SUFFIX = ".sha256"


class RawModifiedError(ValueError):
    """A raw file would change, or has changed, after it was stored."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_name(value: str, what: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError(f"{what} {value!r} must be a plain folder name")
    return value


def _write_sidecar(target: Path, checksum: str) -> None:
    sidecar = target.with_name(target.name + SIDECAR_SUFFIX)
    sidecar.write_text(f"{checksum}  {target.name}\n", encoding="utf-8")
    sidecar.chmod(READ_ONLY)


def store_raw(source_file: Path, raw_dir: Path, dataset: str, version: str) -> Path:
    """Copy `source_file` into the raw store and return the stored path."""
    if not source_file.is_file():
        raise FileNotFoundError(f"Raw source {source_file} does not exist")
    folder = raw_dir / _plain_name(dataset, "dataset") / _plain_name(version, "version")
    target = folder / source_file.name
    incoming = file_sha256(source_file)

    if target.exists():
        sidecar = target.with_name(target.name + SIDECAR_SUFFIX)
        actual = file_sha256(target)
        recorded = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.exists() else actual
        if actual != recorded:
            raise RawModifiedError(
                f"{target} no longer matches its recorded checksum {recorded}: "
                "raw files must never be edited"
            )
        if actual != incoming:
            raise RawModifiedError(
                f"{target.name} is already stored for {dataset} {version} with different "
                f"content (stored {actual}, new {incoming}). Use a new version folder."
            )
        if not sidecar.exists():
            _write_sidecar(target, actual)
        return target

    folder.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=folder, prefix=".incoming-")
    os.close(handle)
    temp = Path(temp_name)
    try:
        shutil.copyfile(source_file, temp)
        temp.chmod(READ_ONLY)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    _write_sidecar(target, incoming)
    return target
