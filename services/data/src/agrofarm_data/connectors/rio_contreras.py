"""Offline ingestion for the reviewed Rio Contreras rainfall source files (DS-015)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from agrofarm_data import __version__

SourceName = Literal["funcagua-monthly", "cayala-daily", "zona16-daily", "chirps-monthly"]
EXCLUDED_ZONA16_START = pd.Timestamp("2025-05-01")
EXCLUDED_ZONA16_END = pd.Timestamp("2025-07-01")
SOURCE_VERSION = "team-provided-2026-11-15"
ATTRIBUTION = "FUNCAGUA (data provided to AgroFarm; use with attribution)"
CHIRPS_ATTRIBUTION = (
    "Climate Hazards Center CHIRPS v3; monthly point extract provided by FUNCAGUA / Meteo Tech"
)


class RioContrerasError(ValueError):
    """A source file does not match the reviewed Rio Contreras data layout."""


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    sheet: str | None
    header: int
    rainfall_column: str
    date_column: str
    time_step: str
    spatial_resolution: str
    coordinates: dict[str, float] | None
    source: str
    attribution: str
    units: str


SPECS: dict[SourceName, SourceSpec] = {
    "funcagua-monthly": SourceSpec(
        dataset="climate_rio_contreras_watershed_monthly_mean",
        sheet="serie_mensual",
        header=0,
        rainfall_column="rainfall_mean_mm",
        date_column="date",
        time_step="monthly; independent spatial mean for each month, not a monthly accumulation",
        spatial_resolution="mean over the Rio Contreras micro-watershed",
        coordinates=None,
        source="FUNCAGUA, Rio Contreras micro-watershed monthly mean series (team-provided file)",
        attribution=ATTRIBUTION,
        units="mm",
    ),
    "cayala-daily": SourceSpec(
        dataset="climate_cit_uvg_cayala_daily",
        sheet="Cayalá lluvia diaria",
        header=1,
        rainfall_column="rainfall_mm",
        date_column="date",
        time_step="daily; dates with blank rainfall omitted",
        spatial_resolution="single station point",
        coordinates={"latitude": 14.6044722, "longitude": -90.4930659},
        source="CIT UVG Cayalá station (ID 234424211), provided courtesy of Meteo Tech / FUNCAGUA",
        attribution="FUNCAGUA; station data courtesy of Meteo Tech / FUNCAGUA",
        units="mm per day",
    ),
    "zona16-daily": SourceSpec(
        dataset="climate_zona16_daily",
        sheet="Z16 lluvia diaria",
        header=1,
        rainfall_column="rainfall_mm",
        date_column="date",
        time_step="daily; dates with blank rainfall omitted; May-June 2025 excluded",
        spatial_resolution="single station point",
        coordinates={"latitude": 14.62, "longitude": -90.48},
        source="Zona 16 station (ID 234423901), provided courtesy of Meteo Tech / FUNCAGUA",
        attribution="FUNCAGUA; station data courtesy of Meteo Tech / FUNCAGUA",
        units="mm per day",
    ),
    "chirps-monthly": SourceSpec(
        dataset="climate_chirps_cayala_point_monthly",
        sheet=None,
        header=0,
        rainfall_column="rainfall_mm",
        date_column="date",
        time_step="monthly accumulated precipitation",
        spatial_resolution="CHIRPS v3 grid-cell value at the supplied extraction point",
        coordinates={"latitude": 14.604, "longitude": -90.490},
        source="Climate Hazards Center CHIRPS v3 monthly point series (team-provided extract)",
        attribution=CHIRPS_ATTRIBUTION,
        units="mm per month",
    ),
}


def _parse_dates(values: pd.Series, *, monthly: bool) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise RioContrerasError("Source contains a non-empty row with an invalid date")
    if monthly:
        parsed = pd.Series(
            [pd.Timestamp(value.year, value.month, 1) for value in parsed],
            index=values.index,
            dtype="datetime64[ns]",
        )
    return parsed.dt.normalize()


def _clean_rows(
    *,
    date_values: pd.Series,
    rainfall_values: pd.Series,
    rainfall_column: str,
    monthly: bool,
    exclude_zona16_period: bool = False,
) -> pd.DataFrame:
    rainfall = pd.to_numeric(rainfall_values, errors="coerce")
    invalid_nonblank = rainfall_values.notna() & rainfall.isna()
    if invalid_nonblank.any():
        raise RioContrerasError("Rainfall column contains non-numeric values")

    present = rainfall.notna()
    if (date_values[present].isna()).any():
        raise RioContrerasError("A rainfall observation has an invalid or missing date")
    dates = _parse_dates(date_values[present], monthly=monthly)
    values = rainfall[present].astype("float64")
    if (values < 0).any():
        raise RioContrerasError("Rainfall values cannot be negative")

    result = pd.DataFrame({"date": dates.to_numpy(), rainfall_column: values.to_numpy()})
    if exclude_zona16_period:
        excluded = result["date"].ge(EXCLUDED_ZONA16_START) & result["date"].lt(EXCLUDED_ZONA16_END)
        result = result.loc[~excluded].copy()
    result = result.sort_values("date", ignore_index=True)
    if result.empty:
        raise RioContrerasError("No usable rainfall observations remain after cleaning")
    if result["date"].duplicated().any():
        raise RioContrerasError("Source contains duplicate dates")
    return result


def read_funcagua_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="serie_mensual")
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    year_col = columns.get("año", columns.get("anio"))
    month_col = columns.get("mes")
    rainfall_col = columns.get("precip_media")
    if year_col is None or month_col is None or rainfall_col is None:
        raise RioContrerasError("FUNCAGUA sheet must have año/anio, mes, and precip_media columns")
    year = pd.to_numeric(frame[year_col], errors="coerce")
    month = pd.to_numeric(frame[month_col], errors="coerce")
    rainfall = pd.to_numeric(frame[rainfall_col], errors="coerce")
    if (frame[year_col].notna() & year.isna()).any():
        raise RioContrerasError("FUNCAGUA year column contains non-numeric values")
    if (frame[month_col].notna() & month.isna()).any():
        raise RioContrerasError("FUNCAGUA month column contains non-numeric values")
    if (frame[rainfall_col].notna() & rainfall.isna()).any():
        raise RioContrerasError("FUNCAGUA precip_media contains non-numeric values")
    present = year.notna() & month.notna() & rainfall.notna()
    if (year[present] % 1 != 0).any() or (month[present] % 1 != 0).any():
        raise RioContrerasError("FUNCAGUA year and month must be whole numbers")
    if (~month[present].between(1, 12)).any():
        raise RioContrerasError("FUNCAGUA month must be between 1 and 12")
    dates = pd.to_datetime(
        {"year": year[present].astype(int), "month": month[present].astype(int), "day": 1},
        errors="coerce",
    )
    return _clean_rows(
        date_values=dates,
        rainfall_values=rainfall[present],
        rainfall_column="rainfall_mean_mm",
        monthly=True,
    )


def _read_daily_workbook(
    path: Path, source: Literal["cayala-daily", "zona16-daily"]
) -> pd.DataFrame:
    spec = SPECS[source]
    if spec.sheet is None:
        raise RioContrerasError(f"{source} requires an Excel worksheet")
    frame = pd.read_excel(path, sheet_name=spec.sheet, header=spec.header)
    normalized = {str(column).strip().casefold(): str(column) for column in frame.columns}
    date_col = normalized.get("fecha")
    rain_col = normalized.get("lluvia (mm)")
    if date_col is None or rain_col is None:
        raise RioContrerasError(f"{spec.sheet} must contain Fecha and Lluvia (mm) columns")
    result = _clean_rows(
        date_values=pd.to_datetime(frame[date_col], errors="coerce"),
        rainfall_values=frame[rain_col],
        rainfall_column=spec.rainfall_column,
        monthly=False,
        exclude_zona16_period=source == "zona16-daily",
    )
    return result


def read_chirps_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "value_center"}
    if not required.issubset(frame.columns):
        raise RioContrerasError("CHIRPS CSV must contain date and value_center columns")
    return _clean_rows(
        date_values=pd.to_datetime(frame["date"], errors="coerce"),
        rainfall_values=frame["value_center"],
        rainfall_column="rainfall_mm",
        monthly=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RioContrerasConnector:
    """Clean one reviewed Rio Contreras source into its own Parquet dataset."""

    version = SOURCE_VERSION

    def __init__(self, source: SourceName, input_file: Path) -> None:
        self.source = source
        self.input_file = input_file
        self.spec = SPECS[source]
        self.dataset = self.spec.dataset

    def fetch(self) -> list[Path]:
        if not self.input_file.is_file():
            raise FileNotFoundError(f"Input file does not exist: {self.input_file}")
        return [self.input_file]

    def clean(self, raw_files: list[Path]) -> pd.DataFrame:
        if len(raw_files) != 1:
            raise RioContrerasError("Expected exactly one source file")
        if self.source == "funcagua-monthly":
            return read_funcagua_monthly(raw_files[0])
        if self.source in {"cayala-daily", "zona16-daily"}:
            return _read_daily_workbook(raw_files[0], self.source)
        return read_chirps_monthly(raw_files[0])

    def write_clean(self, frame: pd.DataFrame, out_dir: Path, raw_files: list[Path]) -> Path:
        if raw_files is None or len(raw_files) != 1:
            raise RioContrerasError("Expected exactly one raw source file for the manifest")
        out_dir.mkdir(parents=True, exist_ok=True)
        table_path = out_dir / f"{self.dataset}.parquet"
        frame.to_parquet(table_path, index=False, engine="pyarrow")
        manifest = {
            "dataset": self.dataset,
            "file": table_path.name,
            "source": self.spec.source,
            "source_version": self.version,
            "license": "Use permitted with attribution to FUNCAGUA by the project data owner; "
            "no separate license text supplied.",
            "attribution": self.spec.attribution,
            "rows": int(len(frame)),
            "date_range": {
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            },
            "time_step": self.spec.time_step,
            "spatial_resolution": self.spec.spatial_resolution,
            "coordinates": self.spec.coordinates,
            "crs": "EPSG:4326",
            "units": {self.spec.rainfall_column: self.spec.units},
            "missing_value": (
                "Source rows with blank rainfall were omitted; clean output has no null rainfall."
            ),
            "excluded_periods": (
                [
                    {
                        "start": "2025-05-01",
                        "end": "2025-06-30",
                        "reason": "suspect Zona 16 gauge data",
                    }
                ]
                if self.source == "zona16-daily"
                else []
            ),
            "checksum_sha256": _sha256(table_path),
            "source_checksum_sha256": _sha256(raw_files[0]),
            "processing_version": __version__,
        }
        manifest_path = out_dir / f"{self.dataset}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def make_connector(source: str, input_file: Path) -> RioContrerasConnector:
    """Build a connector for one of the explicitly reviewed data sources."""
    if source not in SPECS:
        raise RioContrerasError(f"Unknown source {source!r}; choose one of {', '.join(SPECS)}")
    return RioContrerasConnector(source, input_file)
