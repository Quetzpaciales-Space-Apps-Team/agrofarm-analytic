"""Feature pipeline: one row per month with the rainfall total and the ENSO state."""

import numpy as np
import pandas as pd

ENSO_LAGS = range(0, 7)  # months. The effect of ENSO on local rain arrives late.
FEATURES = ["month_sin", "month_cos"] + [f"roni_lag{lag}" for lag in ENSO_LAGS]
TARGET = "rainfall_mm"


def monthly_rainfall(climate: pd.DataFrame, max_missing_fraction: float = 0.1) -> pd.DataFrame:
    """Monthly rainfall totals from daily values.

    Missing days are never counted as zero rain. A month with a few missing days is scaled
    to its full length. A month with more than `max_missing_fraction` missing has no total.
    """
    daily = climate[["date", TARGET]].copy()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    grouped = daily.groupby("month")[TARGET]
    observed = grouped.count()
    days_in_month = pd.Series(
        pd.DatetimeIndex(observed.index).days_in_month, index=observed.index, dtype="float64"
    )
    missing_fraction = 1.0 - observed / days_in_month
    total = grouped.sum(min_count=1) * days_in_month / observed.where(observed > 0)
    total = total.where(missing_fraction <= max_missing_fraction)
    return pd.DataFrame(
        {"month": observed.index, TARGET: total.to_numpy(), "missing_fraction": missing_fraction}
    ).reset_index(drop=True)


def month_cycle(month_of_year: object) -> tuple[np.ndarray, np.ndarray]:
    """Month of year (1 to 12) as a point on a circle, so December sits next to January."""
    angle = 2 * np.pi * (np.asarray(month_of_year, dtype=float) - 1) / 12.0
    return np.sin(angle), np.cos(angle)


def build_training_table(climate: pd.DataFrame, enso: pd.DataFrame) -> pd.DataFrame:
    """Join monthly rainfall with lagged RONI. Rows with any missing value are dropped."""
    table = monthly_rainfall(climate)[["month", TARGET]]
    roni = enso.set_index("month")["roni"].sort_index()
    for lag in ENSO_LAGS:
        lagged = roni.copy()
        lagged.index = lagged.index + pd.DateOffset(months=lag)
        table[f"roni_lag{lag}"] = table["month"].map(lagged)
    table["year"] = table["month"].dt.year
    table["month_of_year"] = table["month"].dt.month
    sin, cos = month_cycle(table["month_of_year"].to_numpy())
    table["month_sin"], table["month_cos"] = sin, cos
    table = table.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)
    return table[["month", "year", "month_of_year", *FEATURES, TARGET]]
