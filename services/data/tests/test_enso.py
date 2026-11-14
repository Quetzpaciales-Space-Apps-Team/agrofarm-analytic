"""ENSO connector: RONI and ONI from NOAA CPC, one row per month with its phase.

The fixtures are short recorded excerpts of the real NOAA files (1997-1999 and 2015).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from agrofarm_data.cli import main
from agrofarm_data.connectors.enso import (
    COLUMNS,
    EnsoError,
    classify_phase,
    read_enso,
    write_clean,
)

FIXTURES = Path(__file__).parent / "fixtures"
ONI = FIXTURES / "oni_sample.txt"
RONI = FIXTURES / "roni_sample.txt"


def month(frame: pd.DataFrame, iso: str) -> pd.Series:  # type: ignore[type-arg]
    return frame.loc[frame["month"] == pd.Timestamp(iso)].iloc[0]


@pytest.mark.parametrize(
    ("value", "phase"),
    [
        (0.5, "el_nino"), (0.51, "el_nino"), (2.4, "el_nino"),
        (-0.5, "la_nina"), (-0.51, "la_nina"), (-1.7, "la_nina"),
        (0.49, "neutral"), (-0.49, "neutral"), (0.0, "neutral"),
    ],
)  # fmt: skip
def test_phase_thresholds_are_inclusive_at_half_a_degree(value: float, phase: str) -> None:
    assert classify_phase(value) == phase


def test_missing_index_has_no_phase() -> None:
    assert classify_phase(float("nan")) is None


def test_each_season_is_assigned_to_its_centre_month() -> None:
    frame = read_enso(ONI, RONI)

    assert list(frame.columns) == COLUMNS
    # DJF 1997 is December 1996 to February 1997: centre month January 1997.
    assert frame["month"].min() == pd.Timestamp("1997-01-01")
    # NDJ 1999 is November 1999 to January 2000: centre month December 1999.
    assert pd.Timestamp("1999-12-01") in set(frame["month"])
    assert frame["month"].is_unique
    assert frame["month"].is_monotonic_increasing


def test_golden_values_from_the_recorded_files() -> None:
    frame = read_enso(ONI, RONI)
    oni_lines = {tuple(line.split()[:2]): line.split() for line in ONI.read_text().splitlines()[1:]}
    roni_lines = {
        tuple(line.split()[:2]): line.split() for line in RONI.read_text().splitlines()[1:]
    }

    # OND 1997: centre month November 1997, the peak of the 1997-98 El Nino.
    row = month(frame, "1997-11-01")
    assert row["oni"] == float(oni_lines[("OND", "1997")][3])
    assert row["roni"] == float(roni_lines[("OND", "1997")][2])
    assert row["phase"] == "el_nino"
    # JAS 1999: centre month August 1999, La Nina.
    assert month(frame, "1999-08-01")["phase"] == "la_nina"


def test_phase_follows_roni_and_the_oni_phase_is_kept_alongside() -> None:
    frame = read_enso(ONI, RONI)

    for _, row in frame.iterrows():
        assert row["phase"] == classify_phase(row["roni"])
        assert row["phase_oni"] == classify_phase(row["oni"])


def test_noaa_missing_marker_becomes_null(tmp_path: Path) -> None:
    roni = tmp_path / "roni.txt"
    roni.write_text("SEAS   YR  ANOM\nDJF  1997 -99.90\nJFM  1997 -0.30\n", encoding="utf-8")
    oni = tmp_path / "oni.txt"
    oni.write_text(
        " SEAS  YR   TOTAL   ANOM\n  DJF 1997  26.0  -0.50\n  JFM 1997  26.5  -0.40\n",
        encoding="utf-8",
    )

    frame = read_enso(oni, roni)

    assert bool(frame["roni"].isna().iloc[0])
    assert frame["phase"].iloc[0] is None
    assert frame["phase_oni"].iloc[0] == "la_nina"


def test_unknown_season_code_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("SEAS   YR  ANOM\nXYZ  1997 -0.30\n", encoding="utf-8")

    with pytest.raises(EnsoError, match="XYZ"):
        read_enso(ONI, bad)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("SEAS   YR  ANOM\n", encoding="utf-8")

    with pytest.raises(EnsoError, match="no rows"):
        read_enso(ONI, empty)


def test_write_clean_produces_parquet_and_a_manifest(tmp_path: Path) -> None:
    manifest_path = write_clean(read_enso(ONI, RONI), tmp_path, oni_file=ONI, roni_file=RONI)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pd.read_parquet(tmp_path / manifest["file"])
    assert manifest["dataset"] == "enso_monthly"
    assert len(table) == manifest["rows"] == 48
    assert manifest["threshold_c"] == 0.5
    assert manifest["phase_index"] == "roni"
    assert manifest["date_range"] == {"start": "1997-01-01", "end": "2015-12-01"}
    assert len(manifest["checksum_sha256"]) == 64


def test_cli_runs_the_connector(tmp_path: Path) -> None:
    assert main(["enso", "--oni", str(ONI), "--roni", str(RONI), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "enso_monthly.parquet").is_file()
    assert (tmp_path / "enso_monthly.manifest.json").is_file()
