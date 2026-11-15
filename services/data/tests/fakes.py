"""A small fake connector used to test the interface, the raw store and `run_job`."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agrofarm_data import __version__


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_manifest() -> dict[str, Any]:
    """A complete manifest for a CC BY table with one raw input."""
    return {
        "dataset": "fake_daily",
        "file": "fake_daily.parquet",
        "source": "Fake provider, https://example.org",
        "source_version": "v1",
        "license": "CC BY 4.0",
        "attribution": "Fake provider",
        "date_range": {"start": "2024-01-01", "end": "2024-01-03"},
        "time_step": "daily",
        "spatial_resolution": "single point",
        "checksum_sha256": "a" * 64,
        "source_checksum_sha256": {"fake.csv": "b" * 64},
        "processing_version": "0.1.0",
    }


class FakeConnector:
    """Reads a three-row CSV and writes a Parquet table with a manifest."""

    dataset = "fake_daily"
    version = "v1"

    def __init__(self, source: Path) -> None:
        self.source = source
        self.fetch_calls = 0

    def fetch(self) -> list[Path]:
        self.fetch_calls += 1
        return [self.source]

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        frame = pd.read_csv(raw_files[0], parse_dates=["date"])
        return frame.drop_duplicates("date").sort_values("date", ignore_index=True)

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        table = out_dir / f"{self.dataset}.parquet"
        frame.to_parquet(table, index=False, engine="pyarrow")
        manifest = valid_manifest() | {
            "date_range": {
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            },
            "checksum_sha256": sha256(table),
            "source_checksum_sha256": {path.name: sha256(path) for path in raw_files},
            "processing_version": __version__,
        }
        path = out_dir / f"{self.dataset}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path


def make_source(folder: Path, name: str = "fake.csv") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("date,value\n2024-01-02,2\n2024-01-01,1\n2024-01-03,3\n", encoding="utf-8")
    return path
