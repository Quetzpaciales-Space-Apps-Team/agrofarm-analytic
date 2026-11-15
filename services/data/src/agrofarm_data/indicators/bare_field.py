"""Bare-field candidates from a low-NDVI series (DS-056).

Input is the clean NDVI table the Planetary Computer connector writes (one row per composite:
`date`, `ndvi_mean`, optionally `date_end`, `status`, `polygon_id`). A row without a usable value
(null NDVI, or a `status` other than "ok") is **unknown**. It is never read as bare soil and it
cuts a run of low observations in two, so a cloudy stretch cannot invent or stretch a period.

The threshold is a provisional screening default, not a crop limit, and it has not been reviewed
by an agronomist (see `BareFieldThresholds`). NDVI is satellite greenness: low NDVI means little
green cover, which can also be a harvested field, a burn or a water body. Treat a period as a
prompt to check the field, not as a finding.
"""

from dataclasses import dataclass

import pandas as pd

POLYGON = "polygon_id"
DEFAULT_POLYGON = "all"
PERIOD_COLUMNS = [POLYGON, "start", "end", "observations", "ndvi_mean", "ndvi_min"]
MONTH_COLUMNS = [POLYGON, "month", "years_observed", "years_bare", "frequency"]


@dataclass(frozen=True)
class BareFieldThresholds:
    """Provisional screening thresholds, adjustable for local review.

    `bare_ndvi_below`: an observation is low when its NDVI is under this value (0.25, because
    bare soil and dry stubble sit around 0.05 to 0.2 and sparse green cover rises above 0.3; this
    is a literature rule of thumb, not measured for Guatemala).
    `min_observations`: consecutive low observations needed for a period. With 16-day composites
    two observations mean about a month, which filters out one noisy scene.
    """

    bare_ndvi_below: float = 0.25
    min_observations: int = 2


def _prepared(series: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("date", "ndvi_mean") if column not in series.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    table = series.copy()
    if POLYGON not in table.columns:
        table[POLYGON] = DEFAULT_POLYGON
    table["date"] = pd.to_datetime(table["date"])
    values = pd.to_numeric(table["ndvi_mean"], errors="coerce")
    if ((values < -1.0) | (values > 1.0)).any():
        raise ValueError("ndvi_mean must be between -1 and 1; a sentinel value was not removed")
    if table.duplicated([POLYGON, "date"]).any():
        raise ValueError("duplicate dates for one polygon")
    usable = values.notna()
    if "status" in table.columns:
        usable &= table["status"].eq("ok")
    table["ndvi_mean"] = values.where(usable)
    return table.sort_values([POLYGON, "date"]).reset_index(drop=True)


def detect_bare_periods(
    series: pd.DataFrame, thresholds: BareFieldThresholds = BareFieldThresholds()
) -> pd.DataFrame:
    """Runs of at least `min_observations` consecutive low observations, per polygon.

    `start` is the date of the first low observation and `end` the end date of the last one
    (`date_end` when the table has it). An unknown observation ends the run.
    """
    table = _prepared(series)
    rows: list[dict[str, object]] = []
    for polygon, group in table.groupby(POLYGON, sort=True):
        run: list[pd.Series] = []

        def close(run: list[pd.Series], polygon: object = polygon) -> None:
            if len(run) < thresholds.min_observations:
                return
            values = pd.Series([item["ndvi_mean"] for item in run])
            last = run[-1]
            end = (
                last["date_end"]
                if "date_end" in last and pd.notna(last["date_end"])
                else last["date"]
            )
            rows.append(
                {
                    POLYGON: polygon,
                    "start": run[0]["date"],
                    "end": pd.Timestamp(end),
                    "observations": len(run),
                    "ndvi_mean": float(values.mean()),
                    "ndvi_min": float(values.min()),
                }
            )

        for _, row in group.iterrows():
            low = pd.notna(row["ndvi_mean"]) and row["ndvi_mean"] < thresholds.bare_ndvi_below
            if low:
                run.append(row)
            else:
                close(run)
                run = []
        close(run)
    return pd.DataFrame(rows, columns=PERIOD_COLUMNS)


def bare_month_frequency(
    series: pd.DataFrame, thresholds: BareFieldThresholds = BareFieldThresholds()
) -> pd.DataFrame:
    """For each calendar month, the share of observed years in which the field was bare.

    A year counts as observed for a month only if that month has at least one usable
    observation, and as bare if the mean of those observations is under the threshold. A month
    with no usable observation in a year is unknown for that year and counts in neither number.
    """
    table = _prepared(series).dropna(subset=["ndvi_mean"])
    table = table.assign(year=table["date"].dt.year, month=table["date"].dt.month)
    monthly = table.groupby([POLYGON, "month", "year"], as_index=False)["ndvi_mean"].mean()
    monthly["bare"] = monthly["ndvi_mean"] < thresholds.bare_ndvi_below
    summary = monthly.groupby([POLYGON, "month"], as_index=False).agg(
        years_observed=("year", "nunique"), years_bare=("bare", "sum")
    )
    summary["years_bare"] = summary["years_bare"].astype(int)
    summary["frequency"] = summary["years_bare"] / summary["years_observed"]
    return pd.DataFrame(summary[MONTH_COLUMNS]).reset_index(drop=True)
