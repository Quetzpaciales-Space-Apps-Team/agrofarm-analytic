import pandas as pd
import pytest

from agrofarm_ml.features import build_training_table
from agrofarm_ml.splits import (
    SplitError,
    assert_whole_year_split,
    choose_holdout_years,
    season_phase_by_year,
    split_by_years,
)


def test_each_year_gets_the_phase_of_its_rainy_season(enso: pd.DataFrame) -> None:
    phases = season_phase_by_year(enso)

    assert set(phases.unique()) == {"el_nino", "la_nina", "neutral"}
    rainy = enso[(enso["month"].dt.year == 1990) & enso["month"].dt.month.between(5, 10)]
    expected = "el_nino" if rainy["roni"].mean() >= 0.5 else "other"
    assert (phases[1990] == "el_nino") == (expected == "el_nino")


def test_holdout_has_at_least_one_el_nino_year_and_one_la_nina_year(enso: pd.DataFrame) -> None:
    phases = season_phase_by_year(enso)

    holdout = choose_holdout_years(phases, years=range(1990, 2021))

    held = {phases[year] for year in holdout}
    assert {"el_nino", "la_nina"} <= held
    assert holdout == sorted(holdout)
    assert choose_holdout_years(phases, years=range(1990, 2021)) == holdout


def test_holdout_fails_when_a_phase_is_missing() -> None:
    phases = pd.Series({2000: "neutral", 2001: "neutral", 2002: "el_nino"})

    with pytest.raises(SplitError, match="la_nina"):
        choose_holdout_years(phases, years=range(2000, 2003))


def test_split_keeps_whole_years_together(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    table = build_training_table(climate, enso)

    train, test = split_by_years(table, holdout_years=[2001, 2010])

    assert set(test["year"]) == {2001, 2010}
    assert not set(train["year"]) & set(test["year"])
    assert len(train) + len(test) == len(table)


def test_a_random_row_split_is_rejected(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    table = build_training_table(climate, enso)
    shuffled = table.sample(frac=1.0, random_state=0)
    train, test = shuffled.iloc[:300], shuffled.iloc[300:]

    with pytest.raises(SplitError, match="both"):
        assert_whole_year_split(train, test)
