"""Read a clean table that data-service published: read-only, by manifest, checksum checked.

This service never reads `raw/` and never writes next to the data. It reads the manifest JSON
(`<dataset>.manifest.json`) and the Parquet file it names. No code is shared with data-service.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class CleanTableError(ValueError):
    """The manifest or the table it names cannot be trusted."""


def read_clean_table(directory: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = directory / f"{dataset}.manifest.json"
    if not manifest_path.is_file():
        raise CleanTableError(f"No manifest for {dataset} in {directory}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_name = str(manifest.get("file", ""))
    table_path = (directory / file_name).resolve()
    if directory.resolve() not in table_path.parents:
        raise CleanTableError(f"{dataset}: the manifest points outside its folder")
    if not table_path.is_file():
        raise CleanTableError(f"{dataset}: the file {file_name} named by the manifest is missing")
    if hashlib.sha256(table_path.read_bytes()).hexdigest() != manifest.get("checksum_sha256"):
        raise CleanTableError(f"{dataset}: checksum does not match the manifest")
    return pd.read_parquet(table_path), manifest
