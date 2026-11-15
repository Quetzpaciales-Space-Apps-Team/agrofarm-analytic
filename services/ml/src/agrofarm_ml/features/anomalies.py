"""Anomalies against a per-cell climatology (mean of the same calendar month)."""

from collections.abc import Iterable, Sequence

import pandas as pd


def _reference(table: pd.DataFrame, reference_years: Iterable[int] | None) -> pd.DataFrame:
    if reference_years is None:
        return table
    return table[table["year"].isin(list(reference_years))]


def climatology(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    by: Sequence[str] = ("cell_id",),
    reference_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Mean of each column per cell and calendar month. NaN values are skipped, never zeros.
    Pass the training years as `reference_years` so the held-out years do not leak in."""
    keys = [*by, "month_of_year"]
    return _reference(table, reference_years).groupby(keys)[list(columns)].mean().reset_index()


def add_anomalies(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    by: Sequence[str] = ("cell_id",),
    reference_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Add `<column>_anomaly` = value minus the climatology of its cell and calendar month.
    A missing value, or a cell and month with no reference value, gives NaN."""
    normals = climatology(table, columns, by=by, reference_years=reference_years)
    normals = normals.rename(columns={c: f"{c}__normal" for c in columns})
    merged = table.merge(normals, on=[*by, "month_of_year"], how="left", validate="many_to_one")
    for column in columns:
        merged[f"{column}_anomaly"] = merged[column] - merged.pop(f"{column}__normal")
    return merged
