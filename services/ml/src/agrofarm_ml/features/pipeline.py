"""The training table: one row per cell and month joining climate, ENSO lags and static layers.

Label independent: nothing here knows what will be predicted. Missing values stay NaN and the
row is kept. Cropland fraction is carried as a weight column and is never used as a mask.
"""

from collections.abc import Iterable, Sequence

import pandas as pd

from agrofarm_ml.features import ENSO_LAGS, month_cycle
from agrofarm_ml.features.anomalies import add_anomalies
from agrofarm_ml.features.validation import validate_climate, validate_enso, validate_static

SUM_COLUMNS = ("rainfall_mm",)  # totals over the month. Everything else is a monthly mean.
STATION = "station"


def monthly_values(
    climate: pd.DataFrame, value_columns: Sequence[str], max_missing_fraction: float = 0.1
) -> pd.DataFrame:
    """Daily values to monthly per cell. A few missing days are tolerated (a total is scaled to
    the full month); a month with more than `max_missing_fraction` missing has NaN."""
    daily = climate.copy()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    days = pd.DatetimeIndex(daily["month"]).days_in_month.to_numpy()
    daily["_days"] = days
    frames = []
    for column in value_columns:
        grouped = daily.groupby(["cell_id", "month"])
        observed = grouped[column].count()
        length = grouped["_days"].first()
        missing = 1.0 - observed / length
        agg = (
            grouped[column].sum(min_count=1) * length / observed.where(observed > 0)
            if (column in SUM_COLUMNS)
            else grouped[column].mean()
        )
        frames.append(agg.where(missing <= max_missing_fraction).rename(column))
    return pd.concat(frames, axis=1).reset_index()


def build_cell_period_table(
    climate: pd.DataFrame,
    enso: pd.DataFrame,
    *,
    value_columns: Sequence[str] = SUM_COLUMNS,
    static: pd.DataFrame | None = None,
    reference_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """One row per cell and month with the values, their anomalies and the ENSO lags."""
    validate_climate(climate, value_columns)
    validate_enso(enso)
    if static is not None:
        validate_static(static)
    daily = climate if "cell_id" in climate.columns else climate.assign(cell_id=STATION)
    table = monthly_values(daily, value_columns)
    table["year"] = table["month"].dt.year
    table["month_of_year"] = table["month"].dt.month
    table["month_sin"], table["month_cos"] = month_cycle(table["month_of_year"].to_numpy())
    roni = enso.set_index("month")["roni"].sort_index()
    for lag in ENSO_LAGS:
        lagged = roni.copy()
        lagged.index = lagged.index + pd.DateOffset(months=lag)
        table[f"roni_lag{lag}"] = table["month"].map(lagged)
    table = add_anomalies(table, list(value_columns), reference_years=reference_years)
    if static is not None:
        table = table.merge(static, on="cell_id", how="left", validate="many_to_one")
    return table.sort_values(["cell_id", "month"], ignore_index=True)
