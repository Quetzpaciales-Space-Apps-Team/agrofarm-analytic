from pathlib import Path

import pytest

from agrofarm_data import __version__
from agrofarm_data.cli import main


def test_cli_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_without_a_command_fails_with_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_module_entry_point_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy

    monkeypatch.setattr("sys.argv", ["agrofarm-data", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("agrofarm_data", run_name="__main__")

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_chirps_cli_runs_locally_from_a_recorded_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures/chirps/chirps-v3.0.rnl.2024.01.01.tif"
    out = tmp_path / "clean"
    raw = tmp_path / "raw"
    assert (
        main(
            [
                "chirps",
                "--bbox",
                "-90.60",
                "14.40",
                "-90.40",
                "14.60",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-01",
                "--out",
                str(out),
                "--raw-dir",
                str(raw),
            ],
            chirps_window_reader=lambda _url, _bbox: fixture.read_bytes(),
        )
        == 0
    )
    assert (out / "climate_chirps_daily.manifest.json").is_file()
    assert "16 cells" in capsys.readouterr().out


def test_chirps_cli_requires_buckets_for_cloud_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "chirps",
                "--bbox",
                "-90.6",
                "14.4",
                "-90.4",
                "14.6",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-01",
                "--gcs",
            ],
            gcs_storage=None,
        )
    assert exit_info.value.code == 2
    assert "GCS_BUCKET_RAW" in capsys.readouterr().err


def test_chirps_cli_runs_cloud_job_and_reports_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Uploader:
        def __init__(self) -> None:
            self.buckets: list[str] = []

        def upload_directory(
            self, bucket: str, _prefix: str, _directory: Path, *, immutable: bool
        ) -> None:
            self.buckets.append(bucket)

    fixture = Path(__file__).parent / "fixtures/chirps/chirps-v3.0.rnl.2024.01.01.tif"
    uploader = Uploader()
    assert (
        main(
            [
                "chirps",
                "--bbox",
                "-90.6",
                "14.4",
                "-90.4",
                "14.6",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-01",
                "--gcs",
                "--raw-bucket",
                "raw-bucket",
                "--clean-bucket",
                "clean-bucket",
            ],
            chirps_window_reader=lambda _url, _bbox: fixture.read_bytes(),
            gcs_storage=uploader,
        )
        == 0
    )
    assert uploader.buckets == ["raw-bucket", "clean-bucket"]
    assert "gs://clean-bucket/clean/climate_chirps_daily/2024/01/" in capsys.readouterr().out
