"""DS-056: periods of low NDVI are bare-field candidates. A cloud gap is never bare soil."""

from typing import Any

import pandas as pd
import pytest

from agrofarm_data.indicators.bare_field import (
    BareFieldThresholds,
    bare_month_frequency,
    detect_bare_periods,
)

NAN = float("nan")


def series(
    values: list[float], start: str = "2020-01-01", step_days: int = 16, **extra: Any
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq=f"{step_days}D")
    frame = pd.DataFrame({"date": dates, "ndvi_mean": values})
    for column, value in extra.items():
        frame[column] = value
    return frame


def test_a_run_of_low_observations_is_one_period_with_its_dates_and_values() -> None:
    frame = series([0.7, 0.2, 0.15, 0.1, 0.65])

    periods = detect_bare_periods(frame)

    assert len(periods) == 1
    row = periods.iloc[0]
    assert row["start"] == pd.Timestamp("2020-01-17")
    assert row["end"] == pd.Timestamp("2020-02-18")  # the last low composite: 01-17, 02-02, 02-18
    assert row["observations"] == 3
    assert row["ndvi_mean"] == pytest.approx(0.15)
    assert row["ndvi_min"] == pytest.approx(0.1)


def test_the_threshold_is_documented_provisional_and_adjustable() -> None:
    thresholds = BareFieldThresholds()
    assert thresholds.bare_ndvi_below == 0.25
    assert thresholds.min_observations == 2
    assert "provisional" in (BareFieldThresholds.__doc__ or "").lower()
    frame = series([0.7, 0.3, 0.3, 0.7])
    assert detect_bare_periods(frame).empty
    assert len(detect_bare_periods(frame, BareFieldThresholds(bare_ndvi_below=0.35))) == 1


def test_one_low_observation_is_not_a_period() -> None:
    assert detect_bare_periods(series([0.7, 0.1, 0.7, 0.8])).empty


def test_a_cloud_gap_is_never_reported_as_bare_soil() -> None:
    green_with_gap = series([0.7, 0.7, NAN, NAN, NAN, 0.7, 0.7])
    assert detect_bare_periods(green_with_gap).empty

    # A gap between two low values is not evidence of bare soil either: the run is cut there.
    cut = series([0.1, 0.1, NAN, 0.1, 0.1])
    periods = detect_bare_periods(cut)
    assert list(periods["observations"]) == [2, 2]
    assert periods.iloc[0]["end"] < pd.Timestamp("2020-02-02")  # before the gap date


def test_scenes_the_connector_marked_unusable_are_ignored_even_if_a_value_is_present() -> None:
    frame = series([0.1, 0.1, 0.1], status=["ok", "too_few_valid_pixels", "ok"])
    assert detect_bare_periods(frame).empty  # the middle scene breaks the run


def test_each_polygon_gets_its_own_periods() -> None:
    frame = pd.concat(
        [series([0.1, 0.1, 0.7], polygon_id="a"), series([0.7, 0.7, 0.7], polygon_id="b")]
    )
    periods = detect_bare_periods(frame)
    assert list(periods["polygon_id"]) == ["a"]


def test_the_period_ends_at_the_composite_end_date_when_the_table_has_one() -> None:
    frame = series([0.1, 0.1, 0.7])
    frame["date_end"] = frame["date"] + pd.Timedelta(days=15)
    periods = detect_bare_periods(frame)
    assert periods.iloc[0]["end"] == pd.Timestamp("2020-01-17") + pd.Timedelta(days=15)


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"date": pd.to_datetime(["2020-01-01"])}),
        series([0.1, 1.4]),
        series([0.1, -3000.0]),
    ],
)
def test_bad_tables_are_rejected_not_guessed(bad: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        detect_bare_periods(bad)


def test_duplicate_dates_for_one_polygon_are_rejected() -> None:
    first = series([0.1, 0.1])
    frame = pd.concat([first, first.iloc[[0]]], ignore_index=True)  # the first date, twice
    with pytest.raises(ValueError, match="duplicate"):
        detect_bare_periods(frame)


def year(y: int, month_values: dict[int, float]) -> pd.DataFrame:
    rows = [{"date": pd.Timestamp(y, m, 15), "ndvi_mean": v} for m, v in month_values.items()]
    return pd.DataFrame(rows)


def test_month_frequency_counts_only_years_with_a_usable_observation() -> None:
    low = {12: 0.1, 1: 0.1, 7: 0.8}
    frame = pd.concat(
        [
            year(2020, low),
            year(2021, low),
            year(2022, {12: NAN, 1: 0.1, 7: 0.8}),  # a cloudy December: unknown, not bare
        ]
    )

    table = bare_month_frequency(frame).set_index("month")

    assert table.loc[12, "years_observed"] == 2
    assert table.loc[12, "years_bare"] == 2
    assert table.loc[12, "frequency"] == pytest.approx(1.0)
    assert table.loc[1, "years_observed"] == 3
    assert table.loc[7, "frequency"] == pytest.approx(0.0)


def test_month_frequency_is_a_share_of_observed_years() -> None:
    frame = pd.concat([year(2020, {3: 0.1}), year(2021, {3: 0.7}), year(2022, {3: 0.1})])
    row = bare_month_frequency(frame).set_index("month").loc[3]
    assert row["years_observed"] == 3
    assert row["frequency"] == pytest.approx(2 / 3)
