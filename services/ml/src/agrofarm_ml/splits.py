"""Validation splits by whole years. A random split leaks the season into the test set and
makes the model look better than it is."""

from collections.abc import Iterable

import pandas as pd

THRESHOLD_C = 0.5
RAINY_SEASON = (5, 10)  # May to October
PHASES = ("el_nino", "la_nina", "neutral")


class SplitError(ValueError):
    """The split would not be a fair test."""


def season_phase_by_year(enso: pd.DataFrame) -> "pd.Series[str]":
    """Phase of each year: mean RONI over its rainy season against the 0.5 degree threshold."""
    season = enso[enso["month"].dt.month.between(*RAINY_SEASON)]
    mean = season.groupby(season["month"].dt.year)["roni"].mean().dropna()
    phase = mean.map(
        lambda value: (
            "el_nino" if value >= THRESHOLD_C else "la_nina" if value <= -THRESHOLD_C else "neutral"
        )
    )
    result: pd.Series[str] = phase.rename("phase")
    return result


def choose_holdout_years(
    phases: "pd.Series[str]", years: Iterable[int], per_phase: int = 2
) -> list[int]:
    """The most recent `per_phase` years of each phase. Needs El Nino and La Nina years."""
    available = phases[phases.index.isin(list(years))]
    holdout: list[int] = []
    for phase in PHASES:
        of_phase = sorted(int(year) for year in available.index[available == phase])
        if not of_phase and phase != "neutral":
            raise SplitError(f"No {phase} year available to hold out.")
        holdout += of_phase[-per_phase:]
    return sorted(holdout)


def assert_whole_year_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = sorted(set(train["year"]) & set(test["year"]))
    if shared:
        raise SplitError(f"Years in both train and test: {shared}. Split by whole years.")


def split_by_years(
    table: pd.DataFrame, holdout_years: Iterable[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_test = table["year"].isin(list(holdout_years))
    train, test = table[~in_test].reset_index(drop=True), table[in_test].reset_index(drop=True)
    assert_whole_year_split(train, test)
    return train, test
