"""Pure climate indicators for daily, per-cell observations.

These are configurable screening defaults, not crop-specific agronomic limits:

* A dry day has rainfall below 1 mm. A dry spell lasts at least 5 consecutive
  calendar days. Missing dates and missing rainfall break a spell.
* A heavy-rain day has at least 20 mm of rainfall.
* A hot day has a maximum temperature of at least 35 degrees C.
* A cold night has a minimum temperature of at most 10 degrees C.
* Daily water balance is rainfall minus reference evapotranspiration, in mm.

Monthly totals are null unless every calendar day has the required observation.
The output includes observed-day counts so data gaps remain explicit. Thresholds
and local interpretation need review before use in recommendations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import nan
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    """Provisional daily screening thresholds, adjustable for local review."""

    dry_day_below_mm: float = 1.0
    dry_spell_min_days: int = 5
    heavy_rain_at_least_mm: float = 20.0
    hot_day_at_least_c: float = 35.0
    cold_night_at_most_c: float = 10.0


def dry_spell_days(
    rainfall: pd.Series,
    *,
    threshold_mm: float = 1.0,
    minimum_days: int = 5,
) -> pd.Series:
    """Mark all days in qualifying dry spells; missing values remain unknown."""
    _validate_daily_series(rainfall)
    if threshold_mm < 0 or minimum_days < 1:
        raise ValueError("Dry-spell thresholds must be nonnegative and at least one day")

    result = pd.Series(pd.array([pd.NA] * len(rainfall), dtype="boolean"), index=rainfall.index)
    run: list[int] = []

    def finish_run() -> None:
        if len(run) >= minimum_days:
            result.iloc[run] = True
        run.clear()

    previous_date: pd.Timestamp | None = None
    for position, (index_value, amount) in enumerate(rainfall.items()):
        date = cast(pd.Timestamp, index_value)
        if previous_date is not None and date - previous_date != pd.Timedelta(days=1):
            finish_run()
        previous_date = date
        if pd.isna(amount):
            finish_run()
        elif float(amount) < threshold_mm:
            result.iloc[position] = False
            run.append(position)
        else:
            result.iloc[position] = False
            finish_run()
    finish_run()
    result.name = "dry_spell_day"
    return result


def heavy_rain_days(rainfall: pd.Series, *, threshold_mm: float = 20.0) -> pd.Series:
    """Flag days with rainfall at or above the heavy-rain threshold."""
    return _daily_events(rainfall, lambda value: value >= threshold_mm, "heavy_rain_day")


def hot_days(maximum_temperature: pd.Series, *, threshold_c: float = 35.0) -> pd.Series:
    """Flag days with maximum temperature at or above the hot-day threshold."""
    return _daily_events(maximum_temperature, lambda value: value >= threshold_c, "hot_day")


def cold_nights(minimum_temperature: pd.Series, *, threshold_c: float = 10.0) -> pd.Series:
    """Flag nights with minimum temperature at or below the cold-night threshold."""
    return _daily_events(minimum_temperature, lambda value: value <= threshold_c, "cold_night")


def water_balance(rainfall: pd.Series, evapotranspiration: pd.Series) -> pd.Series:
    """Return daily rainfall minus reference evapotranspiration, preserving gaps."""
    _validate_daily_series(rainfall)
    _validate_daily_series(evapotranspiration)
    if not rainfall.index.equals(evapotranspiration.index):
        raise ValueError("Rainfall and evapotranspiration dates must match")
    result = rainfall.astype("float64") - evapotranspiration.astype("float64")
    result.name = "water_balance_mm"
    return result


def summarize_monthly(
    frame: pd.DataFrame, *, thresholds: Thresholds = Thresholds()
) -> pd.DataFrame:
    """Summarize daily indicators for each cell and calendar month.

    An indicator is null when any required day is absent or has a missing value.
    Spells are evaluated across month boundaries before monthly counting. If
    evapotranspiration is unavailable, water balance is null but the other
    indicators are still produced.
    """
    required = [
        "date",
        "latitude",
        "longitude",
        "rainfall_mm",
        "temperature_max_c",
        "temperature_min_c",
    ]
    _require_columns(frame, required)
    optional = ["evapotranspiration_mm"] if "evapotranspiration_mm" in frame else []
    data = frame[[*required, *optional]].copy()
    if not optional:
        data["evapotranspiration_mm"] = nan
    data["date"] = pd.to_datetime(data["date"])
    if data[["date", "latitude", "longitude"]].isna().any().any():
        raise ValueError("Dates and coordinates cannot be missing")
    if data.duplicated(["latitude", "longitude", "date"]).any():
        raise ValueError("Each cell may have only one observation per day")
    data = data.sort_values(["latitude", "longitude", "date"])

    rows: list[dict[str, object]] = []
    for (latitude, longitude), cell in data.groupby(["latitude", "longitude"], sort=True):
        daily = cell.set_index("date")
        rain = daily["rainfall_mm"]
        maximum = daily["temperature_max_c"]
        minimum = daily["temperature_min_c"]
        evaporation = daily["evapotranspiration_mm"]
        dry = dry_spell_days(
            rain,
            threshold_mm=thresholds.dry_day_below_mm,
            minimum_days=thresholds.dry_spell_min_days,
        )
        heavy = heavy_rain_days(rain, threshold_mm=thresholds.heavy_rain_at_least_mm)
        hot = hot_days(maximum, threshold_c=thresholds.hot_day_at_least_c)
        cold = cold_nights(minimum, threshold_c=thresholds.cold_night_at_most_c)
        balance = water_balance(rain, evaporation)
        dates = pd.DatetimeIndex(daily.index)
        months = pd.period_range(dates.min(), dates.max(), freq="M")
        for month in months:
            in_month = dates.to_period("M") == month
            days = month.days_in_month
            rain_observed = int(rain.loc[in_month].count())
            max_observed = int(maximum.loc[in_month].count())
            min_observed = int(minimum.loc[in_month].count())
            balance_observed = int(balance.loc[in_month].count())
            rows.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "month": month.to_timestamp(),
                    "rainfall_observed_days": rain_observed,
                    "temperature_max_observed_days": max_observed,
                    "temperature_min_observed_days": min_observed,
                    "water_balance_observed_days": balance_observed,
                    "dry_spell_days": int(dry.loc[in_month].sum())
                    if rain_observed == days
                    else nan,
                    "heavy_rain_days": int(heavy.loc[in_month].sum())
                    if rain_observed == days
                    else nan,
                    "hot_days": int(hot.loc[in_month].sum()) if max_observed == days else nan,
                    "cold_nights": int(cold.loc[in_month].sum()) if min_observed == days else nan,
                    "water_balance_mm": float(balance.loc[in_month].sum())
                    if balance_observed == days
                    else nan,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "latitude",
            "longitude",
            "month",
            "rainfall_observed_days",
            "temperature_max_observed_days",
            "temperature_min_observed_days",
            "water_balance_observed_days",
            "dry_spell_days",
            "heavy_rain_days",
            "hot_days",
            "cold_nights",
            "water_balance_mm",
        ],
    )


def add_monthly_normals_and_anomalies(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Add per-cell calendar-month means and deviations across available years."""
    _require_columns(frame, ["latitude", "longitude", "month", *value_columns])
    if not value_columns:
        raise ValueError("At least one value column is required")
    data = frame.copy()
    data["month"] = pd.to_datetime(data["month"])
    if not data["month"].dt.is_month_start.all():
        raise ValueError("month must be the first day of each calendar month")
    data["_calendar_month"] = data["month"].dt.month
    keys = ["latitude", "longitude", "_calendar_month"]
    normals = data.groupby(keys, dropna=False)[list(value_columns)].mean()
    normals = normals.rename(columns={column: f"{column}_normal" for column in value_columns})
    result = data.join(normals, on=keys)
    for column in value_columns:
        result[f"{column}_anomaly"] = result[column] - result[f"{column}_normal"]
    return result.drop(columns="_calendar_month")


def _daily_events(series: pd.Series, predicate: Callable[[float], bool], name: str) -> pd.Series:
    _validate_daily_series(series)
    flags = [pd.NA if pd.isna(value) else predicate(float(value)) for value in series]
    return pd.Series(pd.array(flags, dtype="boolean"), index=series.index, name=name)


def _validate_daily_series(series: pd.Series) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Daily observations need a DatetimeIndex")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("Daily dates must be unique and sorted")
    if not (series.index == series.index.normalize()).all():
        raise ValueError("Daily dates must be midnight calendar dates")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
