"""Climate indicator rules on synthetic daily observations."""

import pandas as pd
import pytest

from agrofarm_data.indicators import (
    Thresholds,
    add_monthly_normals_and_anomalies,
    cold_nights,
    dry_spell_days,
    heavy_rain_days,
    hot_days,
    summarize_monthly,
    water_balance,
)


def test_dry_spell_requires_five_consecutive_observed_dry_days() -> None:
    rain = pd.Series(
        [0.0, 0.5, 0.9, 0.0, 0.0, 2.0],
        index=pd.date_range("2024-01-01", periods=6),
    )

    assert dry_spell_days(rain).tolist() == [True, True, True, True, True, False]


def test_missing_calendar_day_and_missing_rain_break_dry_spells() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06"])
    rain = pd.Series([0.0, 0.0, 0.0, 0.0, None], index=dates)

    result = dry_spell_days(rain)

    assert result.iloc[:4].tolist() == [False] * 4
    assert pd.isna(result.iloc[4])


def test_dry_spell_can_cross_a_month_boundary() -> None:
    rain = pd.Series(0.0, index=pd.date_range("2024-01-29", periods=5))

    assert dry_spell_days(rain).tolist() == [True] * 5


def test_event_thresholds_are_inclusive_and_missing_values_stay_unknown() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    rain = pd.Series([19.9, 20.0, None], index=dates)
    maximum = pd.Series([34.9, 35.0, None], index=dates)
    minimum = pd.Series([10.1, 10.0, None], index=dates)

    for result in (heavy_rain_days(rain), hot_days(maximum), cold_nights(minimum)):
        assert not bool(result.iloc[0])
        assert bool(result.iloc[1])
        assert pd.isna(result.iloc[2])


def test_water_balance_subtracts_evapotranspiration_and_propagates_missing() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    rain = pd.Series([5.0, None, 2.0], index=dates)
    evaporation = pd.Series([2.0, 1.0, None], index=dates)

    result = water_balance(rain, evaporation)

    assert result.iloc[0] == pytest.approx(3.0)
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_monthly_summary_counts_events_and_water_balance_for_a_complete_month() -> None:
    dates = pd.date_range("2024-01-01", "2024-01-31")
    frame = pd.DataFrame(
        {
            "date": dates,
            "latitude": 14.5,
            "longitude": -90.5,
            "rainfall_mm": 0.0,
            "temperature_max_c": 20.0,
            "temperature_min_c": 12.0,
            "evapotranspiration_mm": 2.0,
        }
    )
    frame.loc[frame["date"].eq(pd.Timestamp("2024-01-16")), "rainfall_mm"] = 20.0
    frame.loc[frame["date"].eq(pd.Timestamp("2024-01-03")), "temperature_max_c"] = 35.0
    frame.loc[frame["date"].eq(pd.Timestamp("2024-01-04")), "temperature_min_c"] = 10.0

    result = summarize_monthly(frame).iloc[0]

    assert result["month"] == pd.Timestamp("2024-01-01")
    assert result["dry_spell_days"] == 30
    assert result["heavy_rain_days"] == 1
    assert result["hot_days"] == 1
    assert result["cold_nights"] == 1
    assert result["water_balance_mm"] == pytest.approx(-42.0)
    assert result["rainfall_observed_days"] == 31


def test_missing_day_makes_monthly_rain_indicators_unknown() -> None:
    dates = pd.date_range("2024-02-01", "2024-02-29").difference(pd.DatetimeIndex(["2024-02-15"]))
    frame = pd.DataFrame(
        {
            "date": dates,
            "latitude": 14.5,
            "longitude": -90.5,
            "rainfall_mm": 0.0,
            "temperature_max_c": 20.0,
            "temperature_min_c": 12.0,
            "evapotranspiration_mm": 2.0,
        }
    )

    result = summarize_monthly(frame).iloc[0]

    assert result["rainfall_observed_days"] == 28
    assert pd.isna(result["dry_spell_days"])
    assert pd.isna(result["heavy_rain_days"])
    assert pd.isna(result["water_balance_mm"])


def test_monthly_indicators_work_without_evapotranspiration_input() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", "2024-01-31"),
            "latitude": 14.5,
            "longitude": -90.5,
            "rainfall_mm": 0.0,
            "temperature_max_c": 20.0,
            "temperature_min_c": 12.0,
        }
    )

    result = summarize_monthly(frame).iloc[0]

    assert result["dry_spell_days"] == 31
    assert result["rainfall_observed_days"] == 31
    assert result["water_balance_observed_days"] == 0
    assert pd.isna(result["water_balance_mm"])


def test_monthly_normals_and_anomalies_are_per_cell_and_calendar_month() -> None:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(["2022-01-01", "2023-01-01", "2022-02-01", "2022-01-01"]),
            "latitude": [14.5, 14.5, 14.5, 14.6],
            "longitude": [-90.5, -90.5, -90.5, -90.5],
            "water_balance_mm": [10.0, 30.0, None, 100.0],
        }
    )

    result = add_monthly_normals_and_anomalies(frame, ["water_balance_mm"])

    assert result["water_balance_mm_normal"].iloc[[0, 1, 3]].tolist() == pytest.approx(
        [20.0, 20.0, 100.0]
    )
    assert result["water_balance_mm_anomaly"].iloc[[0, 1, 3]].tolist() == pytest.approx(
        [-10.0, 10.0, 0.0]
    )
    assert pd.isna(result["water_balance_mm_anomaly"].iloc[2])


def test_thresholds_can_be_changed_for_local_review() -> None:
    rain = pd.Series([2.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    thresholds = Thresholds(dry_day_below_mm=3.0, dry_spell_min_days=2)

    assert dry_spell_days(
        rain,
        threshold_mm=thresholds.dry_day_below_mm,
        minimum_days=thresholds.dry_spell_min_days,
    ).tolist() == [True, True]
