"""The connector interface and the idempotent job runner.

A connector is one source: `fetch` gets raw inputs, `clean` turns them into a table,
`write_clean` writes the clean output plus its manifest. `run_job` stores the raw inputs
once (`raw_store`), cleans from the stored copies and checks the manifest.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from agrofarm_data.manifest import validate_manifest
from agrofarm_data.raw_store import store_raw


@runtime_checkable
class Connector(Protocol):
    dataset: str  # clean dataset name, also the raw folder name
    version: str  # source release, the raw folder below the dataset

    def fetch(self) -> list[Path]:
        """Return the raw input files (downloaded, or found locally)."""
        ...

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        """Validate and normalize the raw files into the clean table."""
        ...

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        """Write the clean output into `out_dir` and return the manifest path."""
        ...


def run_job(connector: Connector, raw_dir: Path, out_dir: Path) -> Path:
    """Run a connector end to end. Running it again changes nothing. Returns the manifest path."""
    fetched = connector.fetch()
    if not fetched:
        raise ValueError(f"{connector.dataset}: fetch returned no raw files")
    stored = [store_raw(path, raw_dir, connector.dataset, connector.version) for path in fetched]
    frame = connector.clean(stored)
    manifest_path = connector.write_clean(frame, out_dir, stored)
    validate_manifest(manifest_path)
    return manifest_path
