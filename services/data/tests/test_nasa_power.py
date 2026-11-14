"""NASA POWER connector: header block, YEAR and DOY columns, -999 means missing, wind in m/s."""

import json
from pathlib import Path

import pandas as pd
import pytest

from agrofarm_data.cli import main
from agrofarm_data.connectors.nasa_power import (
    COLUMNS,
    PowerError,
    read_power,
    write_clean,
)

HEADER = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Daily Data 
Dates (month/day/year): 12/30/2024 through 01/02/2025 in LST
Location: latitude  14.59   longitude -90.53 
elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 1156.37 meters
The value for missing source data that cannot be computed or is outside \
of the sources availability range: -999 
parameter(s): 
T2M                   MERRA-2 Temperature at 2 Meters (C) 
WS2M                  MERRA-2 Wind Speed at 2 Meters (m/s) 
-END HEADER-
"""  # noqa: W291
COLUMNS_LINE = "YEAR,DOY,T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M\n"
ROWS = [
    "2024,365,19.5,12.0,29.0,0.4,60.0,16.8,2.0",
    "2024,366,20.0,14.0,28.0,1.5,-999,17.8,-999",
    "2025,1,21.0,15.0,29.5,0.0,65.0,-999,1.0",
    "2025,2,20.5,14.0,29.5,12.25,58.0,20.0,0.5",
]


def make_file(path: Path, rows: list[str] = ROWS, header: str = HEADER,
              columns: str = COLUMNS_LINE) -> Path:  # fmt: skip
    path.write_text(header + columns + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return make_file(tmp_path / "power.csv")


def test_year_and_day_of_year_become_dates_across_a_leap_year(source: Path) -> None:
    frame = read_power(source)
    assert list(frame.columns) == COLUMNS
    assert [day.date().isoformat() for day in frame["date"]] == [
        "2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02",
    ]  # fmt: skip


def test_location_comes_from_the_header(source: Path) -> None:
    frame = read_power(source)
    assert set(frame["latitude"]) == {14.59}
    assert set(frame["longitude"]) == {-90.53}


def test_missing_marker_becomes_null_never_a_number(source: Path) -> None:
    frame = read_power(source)
    assert bool(frame["relative_humidity_pct"].isna().iloc[1])
    assert bool(frame["wind_speed_kmh"].isna().iloc[1])
    assert bool(frame["solar_radiation_mj_m2"].isna().iloc[2])
    assert int(frame.drop(columns=["date"]).isna().sum().sum()) == 3
    assert not (frame.drop(columns=["date"]) == -999).any().any()


def test_missing_marker_is_read_from_the_header(tmp_path: Path) -> None:
    header = HEADER.replace("range: -999", "range: -9999")
    rows = [ROWS[0], "2024,366,20.0,14.0,28.0,-9999,60.0,17.8,1.0"]
    frame = read_power(make_file(tmp_path / "p.csv", rows=rows, header=header))
    assert bool(frame["rainfall_mm"].isna().iloc[1])


def test_wind_is_converted_from_metres_per_second_to_km_per_hour(source: Path) -> None:
    frame = read_power(source)
    assert frame["wind_speed_kmh"].iloc[0] == pytest.approx(7.2)
    assert frame["wind_speed_kmh"].iloc[3] == pytest.approx(1.8)


def test_values_keep_their_units(source: Path) -> None:
    last = read_power(source).iloc[3]
    assert last["temperature_mean_c"] == 20.5
    assert last["temperature_min_c"] == 14.0
    assert last["temperature_max_c"] == 29.5
    assert last["rainfall_mm"] == 12.25
    assert last["relative_humidity_pct"] == 58.0
    assert last["solar_radiation_mj_m2"] == 20.0


@pytest.mark.parametrize(
    ("rows", "header", "columns", "message"),
    [
        (ROWS, HEADER, "YEAR,DOY,T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M\n", "ALLSKY_SFC_SW_DWN"),
        (ROWS, "", COLUMNS_LINE, "header"),
        (ROWS, HEADER.replace("Location: latitude  14.59   longitude -90.53", "Location: ?"),
         COLUMNS_LINE, "Location"),
        ([ROWS[0], ROWS[0]], HEADER, COLUMNS_LINE, "duplicate"),
        ([ROWS[0], ROWS[2]], HEADER, COLUMNS_LINE, "missing days"),
        (["2024,365,19.5,12.0,29.0,-3.0,60.0,16.8,2.0"], HEADER, COLUMNS_LINE, "rainfall_mm"),
        (["2024,365,19.5,22.0,29.0,0.0,60.0,16.8,2.0"], HEADER, COLUMNS_LINE, "minimum"),
        (["2024,365,19.5,12.0,29.0,0.0,160.0,16.8,2.0"], HEADER, COLUMNS_LINE,
         "relative_humidity_pct"),
        ([], HEADER, COLUMNS_LINE, "no data rows"),
    ],
    ids=["column", "no_header", "no_location", "duplicate", "gap", "negative_rain",
         "min_above_mean", "humidity", "empty"],
)  # fmt: skip
def test_broken_files_are_rejected_with_a_clear_message(
    tmp_path: Path, rows: list[str], header: str, columns: str, message: str
) -> None:
    path = make_file(tmp_path / "bad.csv", rows=rows, header=header, columns=columns)
    with pytest.raises(PowerError, match=message):
        read_power(path)


def test_clean_table_and_manifest_are_written(source: Path, tmp_path: Path) -> None:
    frame = read_power(source)
    manifest_path = write_clean(frame, tmp_path / "clean", source_file=source)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    table = pd.read_parquet(tmp_path / "clean" / manifest["file"])

    assert manifest["dataset"] == "climate_nasa_power_daily"
    assert "NASA" in manifest["source"]
    assert manifest["rows"] == len(table) == 4
    assert manifest["date_range"] == {"start": "2024-12-30", "end": "2025-01-02"}
    assert manifest["location"] == {"latitude": 14.59, "longitude": -90.53}
    assert manifest["units"]["wind_speed_kmh"] == "km/h"
    assert manifest["missing_fraction"]["relative_humidity_pct"] == 0.25
    assert len(manifest["checksum_sha256"]) == len(manifest["source_checksum_sha256"]) == 64
    pd.testing.assert_frame_equal(table, frame)


def test_command_line_writes_the_clean_dataset(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clean"
    assert main(["nasa-power", "--input", str(source), "--out", str(out)]) == 0
    assert (out / "climate_nasa_power_daily.parquet").is_file()
    assert "4 rows" in capsys.readouterr().out
