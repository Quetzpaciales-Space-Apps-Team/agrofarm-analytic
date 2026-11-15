"""Spatial and temporal harmonization for clean gridded datasets."""

from collections.abc import Sequence
from math import ceil, floor
from typing import Any, Literal, cast

import pandas as pd

COMMON_RESOLUTION_DEGREES = 0.05
TimeFrequency = Literal["dekad", "month"]
AggregationRule = Literal["sum", "mean", "max"]


def harmonize_spatial(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    resolution_degrees: float = COMMON_RESOLUTION_DEGREES,
) -> pd.DataFrame:
    """Resample a regular grid to aligned cell centers.

    Finer cells are averaged into 0.05-degree cells. Coarser rectangular grids are
    linearly interpolated along latitude and longitude; such output rows are flagged
    with ``spatially_interpolated``. Missing measurements stay missing.
    """
    _require_columns(frame, ["latitude", "longitude", *value_columns])
    if resolution_degrees <= 0:
        raise ValueError("resolution_degrees must be positive")
    if not value_columns:
        raise ValueError("At least one value column is required")
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError("latitude and longitude cannot be missing")

    values = list(value_columns)
    preserved = [
        column
        for column in frame.columns
        if column not in {"latitude", "longitude", *values, "spatially_interpolated"}
    ]
    grouping = preserved
    source_latitudes = sorted(frame["latitude"].dropna().unique().tolist())
    source_longitudes = sorted(frame["longitude"].dropna().unique().tolist())
    spacings = _positive_spacings(source_latitudes) + _positive_spacings(source_longitudes)
    is_coarse = bool(spacings) and min(spacings) > resolution_degrees + 1e-9

    if not is_coarse:
        result = frame.copy()
        result["latitude"] = result["latitude"].map(
            lambda coordinate: _cell_center(float(coordinate), resolution_degrees)
        )
        result["longitude"] = result["longitude"].map(
            lambda coordinate: _cell_center(float(coordinate), resolution_degrees)
        )
        keys = [*grouping, "latitude", "longitude"]
        result = result.groupby(keys, as_index=False, dropna=False)[values].mean()
        result["spatially_interpolated"] = False
        return result[[*grouping, "latitude", "longitude", *values, "spatially_interpolated"]]

    if len(source_latitudes) < 2 or len(source_longitudes) < 2:
        raise ValueError("Coarse-grid interpolation requires at least two coordinates per axis")

    outputs: list[pd.DataFrame] = []
    grouper: str | list[str] = grouping[0] if len(grouping) == 1 else grouping
    partitions = frame.groupby(grouper, dropna=False, sort=False) if grouping else [((), frame)]
    for group_key, part in partitions:
        part_key = group_key if isinstance(group_key, tuple) else (group_key,)
        lats = sorted(part["latitude"].dropna().unique().tolist())
        lons = sorted(part["longitude"].dropna().unique().tolist())
        target_lats = _target_centers(lats[0], lats[-1], resolution_degrees)
        target_lons = _target_centers(lons[0], lons[-1], resolution_degrees)
        if not target_lats or not target_lons:
            continue
        for value in values:
            wide = part.pivot(index="latitude", columns="longitude", values=value).sort_index()
            if wide.isna().all().all():
                interpolated = pd.DataFrame(index=target_lats, columns=target_lons, dtype="float64")
            else:
                wide = wide.reindex(
                    index=sorted(set(lats) | set(target_lats)),
                    columns=sorted(set(lons) | set(target_lons)),
                )
                wide = wide.interpolate(method="index", axis=0, limit_area="inside")
                wide = wide.interpolate(method="index", axis=1, limit_area="inside")
                interpolated = wide.reindex(index=target_lats, columns=target_lons)
            interpolated.index.name = "latitude"
            interpolated.columns.name = "longitude"
            stacked = cast(Any, interpolated.stack(future_stack=True))
            stacked.name = value
            long = stacked.reset_index()
            if not outputs or value == values[0]:
                outputs.append(long)
            else:
                outputs[-1] = outputs[-1].merge(long, on=["latitude", "longitude"], how="outer")
        output = outputs[-1]
        for column, item in zip(grouping, part_key, strict=True):
            output[column] = cast(Any, item)
        output["spatially_interpolated"] = True

    if not outputs:
        return pd.DataFrame(
            columns=[*grouping, "latitude", "longitude", *values, "spatially_interpolated"]
        )
    result = pd.concat(outputs, ignore_index=True)
    return result[[*grouping, "latitude", "longitude", *values, "spatially_interpolated"]]


def aggregate_time(
    frame: pd.DataFrame,
    rules: dict[str, AggregationRule],
    *,
    frequency: TimeFrequency,
    time_column: str = "date",
) -> pd.DataFrame:
    """Aggregate measurements by cell and month or by calendar dekad.

    A dekad is days 1–10, 11–20, or 21–month-end. Missing days are not synthesized
    and all-missing rainfall groups remain null instead of becoming zero.
    """
    _require_columns(frame, [time_column, *rules])
    if not rules:
        raise ValueError("At least one aggregation rule is required")
    if frequency not in ("dekad", "month"):
        raise ValueError("frequency must be 'dekad' or 'month'")
    invalid = {column: rule for column, rule in rules.items() if rule not in ("sum", "mean", "max")}
    if invalid:
        raise ValueError(f"Unsupported aggregation rules: {invalid}")

    data = frame.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    if frequency == "month":
        data[time_column] = data[time_column].dt.to_period("M").dt.to_timestamp()
    else:
        days = data[time_column].dt.day
        start_day = pd.Series(1, index=data.index, dtype="int64")
        start_day.loc[days.between(11, 20)] = 11
        start_day.loc[days.ge(21)] = 21
        offset = pd.to_timedelta(days - start_day, unit="D")
        data[time_column] = data[time_column].dt.normalize() - offset

    group_columns = [
        column
        for column in data.columns
        if column not in {time_column, *rules, "spatially_interpolated"}
    ]
    keys = [*group_columns, time_column]
    grouped = data.groupby(keys, dropna=False, sort=True)
    output = grouped.size().rename("_row_count").reset_index().drop(columns="_row_count")
    for column, rule in rules.items():
        if rule == "sum":
            aggregated = grouped[column].sum(min_count=1).rename(column).reset_index()
        else:
            aggregated = grouped[column].agg(rule).rename(column).reset_index()
        output = output.merge(aggregated, on=keys, how="left", sort=False)
    if "spatially_interpolated" in data:
        flags = (
            grouped["spatially_interpolated"].any().rename("spatially_interpolated").reset_index()
        )
        output = output.merge(flags, on=keys, how="left", sort=False)
    return output[
        [
            *group_columns,
            time_column,
            *rules,
            *(["spatially_interpolated"] if "spatially_interpolated" in output else []),
        ]
    ]


def add_monthly_anomalies(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    cell_columns: Sequence[str] = ("latitude", "longitude"),
    time_column: str = "date",
) -> pd.DataFrame:
    """Append per-cell calendar-month normals and anomalies for each measurement."""
    _require_columns(frame, [time_column, *cell_columns, *value_columns])
    if not value_columns:
        raise ValueError("At least one value column is required")
    data = frame.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    data["_month_of_year"] = data[time_column].dt.month
    keys = [*cell_columns, "_month_of_year"]
    normals = data.groupby(keys, dropna=False, sort=False)[list(value_columns)].mean()
    normals = normals.rename(columns={column: f"{column}_normal" for column in value_columns})
    result = data.join(normals, on=keys)
    for column in value_columns:
        result[f"{column}_anomaly"] = result[column] - result[f"{column}_normal"]
    return result.drop(columns="_month_of_year")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _positive_spacings(coordinates: list[float]) -> list[float]:
    return [
        right - left
        for left, right in zip(coordinates, coordinates[1:], strict=False)
        if right > left
    ]


def _cell_center(coordinate: float, resolution: float) -> float:
    return round(floor(coordinate / resolution) * resolution + resolution / 2, 10)


def _target_centers(minimum: float, maximum: float, resolution: float) -> list[float]:
    first_index = ceil((minimum - resolution / 2) / resolution - 1e-10)
    last_index = floor((maximum - resolution / 2) / resolution + 1e-10)
    return [
        round(index * resolution + resolution / 2, 10)
        for index in range(first_index, last_index + 1)
    ]
