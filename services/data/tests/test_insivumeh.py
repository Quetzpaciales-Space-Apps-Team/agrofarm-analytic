"""INSIVUMEH connector: two sheets, header on row 5, -99.9 means missing."""

import datetime as dt
import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from agrofarm_data.cli import main
from agrofarm_data.connectors.insivumeh import (
    COLUMNS,
    InsivumehError,
    read_insivumeh,
    write_clean,
)

HEADER = [
    "FECHA", "NOMBRE_ESTACIÓN", "TEMPERATURA_MEDIA", "PRECIPITACIÓN",
    "BRILLO_SOLAR", "NUBOSIDAD", "VELOCIDAD_VIENTO", "HUMEDAD_RELATIVA",
]  # fmt: skip
Row = tuple[dt.datetime, str, float, float, float, float, float, float]


def row(day: str, temp: float = 20.0, rain: float = 0.0, sun: float = 6.0,
        cloud: float = 4.0, wind: float = 5.0, hum: float = 75.0) -> Row:  # fmt: skip
    return (dt.datetime.fromisoformat(day), "INSIVUMEH", temp, rain, sun, cloud, wind, hum)


def make_workbook(path: Path, sheets: dict[str, list[Row]], header: list[str] = HEADER) -> Path:
    """Same layout as the real file: three metadata rows, a blank row, header on row 5."""
    book = openpyxl.Workbook()
    book.remove(book.active)  # type: ignore[arg-type]
    for title, rows in sheets.items():
        sheet = book.create_sheet(title)
        sheet.append(["Estación:", "INSIVUMEH", "Periodo:", title])
        sheet.append(["Departamento:", "Guatemala", -99.9, "No existe dato"])
        sheet.append(["Municipio:", "Guatemala", "Nota:", "Datos preliminares"])
        sheet.append([])
        sheet.append(header)
        for data_row in rows:
            sheet.append(list(data_row))
    units = book.create_sheet("DIMENSIONALES")
    units.append(["Variables"])
    book.save(path)
    return path


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    return make_workbook(
        tmp_path / "insivumeh.xlsx",
        {
            "1995-2022": [
                row("1995-01-01", temp=16.7, cloud=-99.9, wind=-99.9, hum=76),
                row("1995-01-02", temp=17.9, rain=12.5),
            ],
            "2023-2025": [row("2023-01-01", temp=21.0, sun=-99.9), row("2023-01-02", rain=-99.9)],
        },
    )


def test_both_sheets_are_merged_in_date_order(workbook: Path) -> None:
    frame = read_insivumeh(workbook)

    assert list(frame.columns) == COLUMNS
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "1995-01-01", "1995-01-02", "2023-01-01", "2023-01-02",
    ]  # fmt: skip


def test_golden_values_keep_their_columns_and_units(workbook: Path) -> None:
    first = read_insivumeh(workbook).iloc[1]

    assert first["station"] == "INSIVUMEH"
    assert first["temperature_mean_c"] == 17.9
    assert first["rainfall_mm"] == 12.5
    assert first["sunshine_hours"] == 6.0
    assert first["cloud_cover_oktas"] == 4.0
    assert first["wind_speed_kmh"] == 5.0
    assert first["relative_humidity_pct"] == 75.0


def test_sentinel_becomes_null_in_every_variable(workbook: Path) -> None:
    frame = read_insivumeh(workbook)
    values = frame[COLUMNS[2:]]

    assert not (values == -99.9).to_numpy().any()
    assert pd.isna(frame.loc[0, "cloud_cover_oktas"])
    assert pd.isna(frame.loc[0, "wind_speed_kmh"])
    assert pd.isna(frame.loc[2, "sunshine_hours"])
    assert pd.isna(frame.loc[3, "rainfall_mm"])
    assert frame.loc[0, "relative_humidity_pct"] == 76


def test_missing_rainfall_is_null_and_never_zero(workbook: Path) -> None:
    frame = read_insivumeh(workbook)

    assert pd.isna(frame.loc[3, "rainfall_mm"])
    assert frame["rainfall_mm"].sum(min_count=1) == 12.5


def test_duplicate_dates_are_rejected(tmp_path: Path) -> None:
    path = make_workbook(
        tmp_path / "dup.xlsx",
        {"1995-2022": [row("1995-01-01")], "2023-2025": [row("1995-01-01")]},
    )

    with pytest.raises(InsivumehError, match="duplicate"):
        read_insivumeh(path)


def test_unexpected_header_is_rejected(tmp_path: Path) -> None:
    path = make_workbook(
        tmp_path / "bad.xlsx", {"1995-2022": [row("1995-01-01")]}, header=HEADER[:-1] + ["OTRA"]
    )

    with pytest.raises(InsivumehError, match="header"):
        read_insivumeh(path)


def test_physically_impossible_values_are_rejected(tmp_path: Path) -> None:
    path = make_workbook(tmp_path / "neg.xlsx", {"1995-2022": [row("1995-01-01", rain=-3.0)]})

    with pytest.raises(InsivumehError, match="rainfall_mm"):
        read_insivumeh(path)


def test_workbook_without_data_sheets_is_rejected(tmp_path: Path) -> None:
    path = make_workbook(tmp_path / "empty.xlsx", {})

    with pytest.raises(InsivumehError, match="no data"):
        read_insivumeh(path)


def test_write_clean_produces_parquet_and_a_manifest(workbook: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"

    manifest_path = write_clean(read_insivumeh(workbook), out, source_file=workbook)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pd.read_parquet(out / manifest["file"])
    assert len(table) == manifest["rows"] == 4
    assert manifest["dataset"] == "climate_insivumeh_daily"
    assert manifest["date_range"] == {"start": "1995-01-01", "end": "2023-01-02"}
    assert manifest["preliminary"] is True
    assert manifest["units"]["rainfall_mm"] == "mm per day"
    assert manifest["units"]["wind_speed_kmh"] == "km/h"
    assert manifest["missing_value"] == "null"
    assert len(manifest["checksum_sha256"]) == 64
    assert len(manifest["source_checksum_sha256"]) == 64
    for key in ("source", "license", "spatial_resolution", "crs", "processing_version"):
        assert manifest[key]


def test_writing_twice_gives_the_same_checksum(workbook: Path, tmp_path: Path) -> None:
    frame = read_insivumeh(workbook)

    first = json.loads(write_clean(frame, tmp_path / "a", source_file=workbook).read_text())
    second = json.loads(write_clean(frame, tmp_path / "b", source_file=workbook).read_text())

    assert first["checksum_sha256"] == second["checksum_sha256"]


def test_cli_runs_the_connector(workbook: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"

    assert main(["insivumeh", "--input", str(workbook), "--out", str(out)]) == 0
    assert (out / "climate_insivumeh_daily.parquet").is_file()
    assert (out / "climate_insivumeh_daily.manifest.json").is_file()
