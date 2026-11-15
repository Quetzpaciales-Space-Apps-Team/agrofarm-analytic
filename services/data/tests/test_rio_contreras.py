"""DS-015 offline import and cleaning rules for the reviewed source files."""

import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from openpyxl.worksheet.worksheet import Worksheet

from agrofarm_data.connectors.rio_contreras import (
    RioContrerasConnector,
    RioContrerasError,
    make_connector,
    read_chirps_monthly,
    read_funcagua_monthly,
)


def make_funcagua(path: Path) -> Path:
    book = openpyxl.Workbook()
    sheet = book.create_sheet("serie_mensual")
    if not isinstance(sheet, Worksheet):
        raise RuntimeError("Expected a worksheet")
    sheet.append(["año", "mes", "fecha", "precip_media"])
    sheet.append([1981, 1, "1981-01-01", 12.5])
    sheet.append([1981, 2, "1981-02-01", 0.0])
    sheet.append([1981, 3, "1981-03-01", None])
    book.save(path)
    return path


def make_station(path: Path, sheet_name: str, data: list[list[object]]) -> Path:
    book = openpyxl.Workbook()
    sheet = book.create_sheet(sheet_name)
    if not isinstance(sheet, Worksheet):
        raise RuntimeError("Expected a worksheet")
    sheet.append(["Daily rainfall source"])
    sheet.append(["Fecha", "Lluvia (mm)", "Mes"])
    for row in data:
        sheet.append(row)
    book.save(path)
    return path


def test_funcagua_keeps_independent_monthly_means_and_drops_blank_rows(
    tmp_path: Path,
) -> None:
    frame = read_funcagua_monthly(make_funcagua(tmp_path / "funcagua.xlsx"))

    assert frame.to_dict("records") == [
        {"date": pd.Timestamp("1981-01-01"), "rainfall_mean_mm": 12.5},
        {"date": pd.Timestamp("1981-02-01"), "rainfall_mean_mm": 0.0},
    ]
    assert not frame.isna().any().any()


def test_cayala_drops_blank_rainfall_but_preserves_zero(tmp_path: Path) -> None:
    source = make_station(
        tmp_path / "cayala.xlsx",
        "Cayalá lluvia diaria",
        [
            ["2024-12-01", 1.2, "2024-12"],
            ["2024-12-02", None, "2024-12"],
            ["2024-12-03", 0.0, "2024-12"],
        ],
    )

    frame = make_connector("cayala-daily", source).clean([source])

    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-12-01", "2024-12-03"]
    assert frame["rainfall_mm"].tolist() == [1.2, 0.0]


def test_zona16_drops_blanks_and_excludes_all_of_may_and_june_2025(tmp_path: Path) -> None:
    source = make_station(
        tmp_path / "zona16.xlsx",
        "Z16 lluvia diaria",
        [
            ["2025-04-30", 2.0, "2025-04"],
            ["2025-05-01", 0.0, "2025-05"],
            ["2025-06-30", 4.0, "2025-06"],
            ["2025-07-01", None, "2025-07"],
            ["2025-07-02", 0.0, "2025-07"],
        ],
    )

    frame = make_connector("zona16-daily", source).clean([source])

    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-04-30", "2025-07-02"]
    assert frame["rainfall_mm"].tolist() == [2.0, 0.0]


def test_chirps_uses_value_center_and_ignores_undefined_summary_columns(tmp_path: Path) -> None:
    source = tmp_path / "chirps.csv"
    source.write_text(
        "date,value_center,mean,min,max\n2021-01,15.9,15.6,13.1,21.2\n",
        encoding="utf-8",
    )

    frame = read_chirps_monthly(source)

    assert frame.to_dict("records") == [{"date": pd.Timestamp("2021-01-01"), "rainfall_mm": 15.9}]


def test_connector_writes_source_specific_manifest_and_parquet(tmp_path: Path) -> None:
    source = make_funcagua(tmp_path / "funcagua.xlsx")
    connector = RioContrerasConnector("funcagua-monthly", source)
    frame = connector.clean([source])

    manifest_path = connector.write_clean(frame, tmp_path / "clean", [source])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pd.read_parquet(manifest_path.parent / manifest["file"])

    assert manifest["dataset"] == "climate_rio_contreras_watershed_monthly_mean"
    assert manifest["time_step"].startswith("monthly;")
    assert manifest["units"] == {"rainfall_mean_mm": "mm"}
    assert manifest["attribution"].startswith("FUNCAGUA")
    assert len(table) == 2
    assert not table.isna().any().any()


def test_unknown_source_fails_with_supported_sources() -> None:
    with pytest.raises(RioContrerasError, match="Unknown source"):
        make_connector("other", Path("source.xlsx"))
