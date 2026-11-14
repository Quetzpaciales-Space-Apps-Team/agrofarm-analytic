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
