"""Input validation for the tables the feature pipeline reads. Fails loudly, names the column.

Missing values are NaN. A sentinel such as -99.9 or -999 means an upstream step broke the
contract with data-service, so it is an error here, never a number.
"""

from collections.abc import Sequence

import pandas as pd

SENTINELS = (-99.9, -999.0, -9999.0)
# Physical limits (same units as data-service: mm per day, degrees C, %, MJ/m2 per day, km/h).
CLIMATE_RANGES = {
    "rainfall_mm": (0.0, 1000.0),
    "temperature_mean_c": (-60.0, 60.0),
    "temperature_min_c": (-60.0, 60.0),
    "temperature_max_c": (-60.0, 60.0),
    "relative_humidity_pct": (0.0, 100.0),
    "solar_radiation_mj_m2": (0.0, 45.0),
    "wind_speed_kmh": (0.0, 300.0),
}
ENSO_RANGES = {"roni": (-5.0, 5.0), "oni": (-5.0, 5.0)}
STATIC_RANGES = {"cropland_fraction": (0.0, 1.0)}


class InputError(ValueError):
    """An input table does not match what the pipeline expects."""


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise InputError(f"{name}: missing columns {', '.join(missing)}")
    if frame.empty:
        raise InputError(f"{name}: the table has no rows")


def _numbers(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise InputError(f"{name}.{column}: must be numeric")
        hits = int(frame[column].isin(SENTINELS).sum())
        if hits:
            raise InputError(
                f"{name}.{column}: {hits} sentinel values ({', '.join(map(str, SENTINELS))}). "
                "Missing values must be NaN."
            )


def _ranges(frame: pd.DataFrame, ranges: dict[str, tuple[float, float]], name: str) -> None:
    for column, (low, high) in ranges.items():
        if column in frame.columns:
            bad = frame[(frame[column] < low) | (frame[column] > high)]
            if not bad.empty:
                raise InputError(
                    f"{name}.{column}: {len(bad)} values outside [{low}, {high}], "
                    f"first {bad[column].iloc[0]}"
                )


def _unique(frame: pd.DataFrame, key: Sequence[str], name: str) -> None:
    count = int(frame.duplicated(list(key)).sum())
    if count:
        raise InputError(f"{name}: {count} duplicate rows for key {list(key)}")


def validate_climate(frame: pd.DataFrame, value_columns: Sequence[str] = ("rainfall_mm",)) -> None:
    _require(frame, ["date", *value_columns], "climate")
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise InputError("climate.date: must be a date column")
    if frame["date"].isna().any():
        raise InputError("climate.date: has missing dates")
    _numbers(frame, value_columns, "climate")
    _ranges(frame, CLIMATE_RANGES, "climate")
    _unique(frame, ["cell_id", "date"] if "cell_id" in frame.columns else ["date"], "climate")


def validate_enso(frame: pd.DataFrame) -> None:
    _require(frame, ["month", "roni"], "enso")
    if not pd.api.types.is_datetime64_any_dtype(frame["month"]):
        raise InputError("enso.month: must be a date column")
    _numbers(frame, [c for c in ENSO_RANGES if c in frame.columns], "enso")
    _ranges(frame, ENSO_RANGES, "enso")
    _unique(frame, ["month"], "enso")


def validate_static(frame: pd.DataFrame) -> None:
    _require(frame, ["cell_id"], "static")
    _numbers(frame, [c for c in STATIC_RANGES if c in frame.columns], "static")
    _ranges(frame, STATIC_RANGES, "static")
    _unique(frame, ["cell_id"], "static")
