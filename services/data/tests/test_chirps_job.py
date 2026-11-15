"""Cloud Run CHIRPS job saves clipped immutable inputs and publishes clean files."""

import json
from datetime import date
from pathlib import Path

from agrofarm_data.chirps_job import run_chirps_job

FIXTURE = Path(__file__).parent / "fixtures/chirps/chirps-v3.0.rnl.2024.01.01.tif"
BBOX = (-90.60, 14.40, -90.40, 14.60)


class FakeUploader:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, list[str], bool]] = []
        self.manifests: list[dict[str, object]] = []

    def upload_directory(
        self, bucket: str, prefix: str, directory: Path, *, immutable: bool
    ) -> None:
        self.uploads.append(
            (
                bucket,
                prefix,
                sorted(
                    path.relative_to(directory).as_posix()
                    for path in directory.rglob("*")
                    if path.is_file()
                ),
                immutable,
            )
        )
        manifest = directory / "climate_chirps_daily.manifest.json"
        if manifest.exists():
            self.manifests.append(json.loads(manifest.read_text()))


def test_cloud_job_reads_a_window_and_uploads_raw_and_clean_products() -> None:
    uploader = FakeUploader()
    result = run_chirps_job(
        bbox=BBOX,
        start=date(2024, 1, 1),
        end=date(2024, 1, 1),
        raw_bucket="agrofarm-raw-stage",
        clean_bucket="agrofarm-clean-stage",
        window_reader=lambda _url, _bbox: FIXTURE.read_bytes(),
        uploader=uploader,
    )

    assert result == (
        "gs://agrofarm-clean-stage/clean/climate_chirps_daily/2024/01/"
        "climate_chirps_daily.manifest.json"
    )
    assert len(uploader.uploads) == 2
    raw, clean = uploader.uploads
    assert raw[0:2] == ("agrofarm-raw-stage", "raw/climate_chirps_daily/v3.0-final-rnl/2024/01")
    assert raw[3] is True
    assert "chirps-v3.0.rnl.2024.01.01.tif" in raw[2]
    assert clean[0:2] == ("agrofarm-clean-stage", "clean/climate_chirps_daily/2024/01")
    assert clean[3] is False
    assert "climate_chirps_daily.manifest.json" in clean[2]
    assert uploader.manifests[0]["rows"] == 16
