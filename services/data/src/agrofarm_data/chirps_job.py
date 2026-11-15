"""Run one monthly CHIRPS ingestion as a Cloud Run Job."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from agrofarm_data.connector import run_job
from agrofarm_data.connectors.chirps import Bbox, ChirpsConnector, WindowReader


class DirectoryUploader(Protocol):
    def upload_directory(
        self, bucket: str, prefix: str, directory: Path, *, immutable: bool
    ) -> None: ...


def run_chirps_job(
    *,
    bbox: Bbox,
    start: date,
    end: date,
    raw_bucket: str,
    clean_bucket: str,
    window_reader: WindowReader,
    uploader: DirectoryUploader,
) -> str:
    """Ingest one calendar month, upload raw window files and publish its clean products."""
    if start > end or (start.year, start.month) != (end.year, end.month):
        raise ValueError("A CHIRPS Cloud Run execution must cover one calendar month")
    partition = f"{start:%Y/%m}"
    with TemporaryDirectory(prefix="agrofarm-chirps-") as work:
        root = Path(work)
        raw_dir, clean_dir = root / "raw", root / "clean"
        connector = ChirpsConnector(
            bbox, start, end, raw_dir / "downloads", window_reader=window_reader
        )
        manifest = run_job(connector, raw_dir, clean_dir)
        raw_source = raw_dir / connector.dataset / connector.version
        clean_prefix = f"clean/{connector.dataset}/{partition}"
        uploader.upload_directory(
            raw_bucket,
            f"raw/{connector.dataset}/{connector.version}/{partition}",
            raw_source,
            immutable=True,
        )
        uploader.upload_directory(clean_bucket, clean_prefix, clean_dir, immutable=False)
    return f"gs://{clean_bucket}/{clean_prefix}/{manifest.name}"
