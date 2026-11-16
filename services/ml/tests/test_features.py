import numpy as np
import pandas as pd

from agrofarm_ml.features import FEATURES, build_training_table, monthly_rainfall


def daily(values: list[float], start: str = "2000-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.date_range(start, periods=len(values), freq="D"), "rainfall_mm": values}
    )


def test_monthly_rainfall_sums_the_days_of_each_month() -> None:
    table = monthly_rainfall(daily([1.0] * 31 + [2.0] * 29))

    assert table["month"].dt.strftime("%Y-%m").tolist() == ["2000-01", "2000-02"]
    assert table["rainfall_mm"].tolist() == [31.0, 58.0]


def test_a_few_missing_days_are_scaled_up_and_never_counted_as_zero_rain() -> None:
    values = [2.0] * 31
    values[3] = float("nan")
    values[17] = float("nan")

    table = monthly_rainfall(daily(values))

    # 29 observed days of 2 mm, scaled to the 31 days of the month.
    assert table["rainfall_mm"].iloc[0] == 62.0
    assert round(float(table["missing_fraction"].iloc[0]), 3) == round(2 / 31, 3)


def test_a_month_with_too_many_missing_days_has_no_total() -> None:
    values = [2.0] * 31
    for day in range(10):
        values[day] = float("nan")

    table = monthly_rainfall(daily(values), max_missing_fraction=0.1)

    assert bool(table["rainfall_mm"].isna().iloc[0])


def test_sentinel_values_never_reach_the_table(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    table = build_training_table(climate, enso)

    assert not (table[FEATURES + ["rainfall_mm"]] == -99.9).to_numpy().any()
    assert not table[FEATURES + ["rainfall_mm"]].isna().to_numpy().any()


def test_enso_lags_point_to_earlier_months(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    table = build_training_table(climate, enso).set_index("month")
    index = enso.set_index("month")["roni"]
    row = table.loc[pd.Timestamp("2005-07-01")]

    assert row["roni_lag0"] == index[pd.Timestamp("2005-07-01")]
    assert row["roni_lag1"] == index[pd.Timestamp("2005-06-01")]
    assert row["roni_lag6"] == index[pd.Timestamp("2005-01-01")]


def test_month_of_year_is_encoded_as_a_cycle(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    table = build_training_table(climate, enso)
    january = table[table["month_of_year"] == 1].iloc[0]
    december = table[table["month_of_year"] == 12].iloc[0]

    assert np.isclose(january["month_sin"] ** 2 + january["month_cos"] ** 2, 1.0)
    # December and January are neighbours on the circle.
    distance = np.hypot(
        january["month_sin"] - december["month_sin"], january["month_cos"] - december["month_cos"]
    )
    assert distance < 0.6


def test_months_without_a_rainfall_total_are_dropped(enso: pd.DataFrame) -> None:
    values = [1.0] * 366
    for day in range(31, 60):  # all of February 2000 is missing
        values[day] = float("nan")

    table = build_training_table(daily(values), enso)

    assert pd.Timestamp("2000-02-01") not in set(table["month"])
    assert len(table) == 11
