"""Harmonization rules checked with small synthetic spatial and time series."""

import pandas as pd
import pytest

from agrofarm_data.harmonize import add_monthly_anomalies, aggregate_time, harmonize_spatial


def test_fine_cells_are_averaged_on_the_common_grid() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [14.501, 14.509, 14.501, 14.509],
            "longitude": [-90.501, -90.501, -90.509, -90.509],
            "rainfall_mm": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = harmonize_spatial(frame, value_columns=["rainfall_mm"])

    assert len(result) == 1
    assert result.iloc[0]["latitude"] == pytest.approx(14.525)
    assert result.iloc[0]["longitude"] == pytest.approx(-90.525)
    assert result.iloc[0]["rainfall_mm"] == pytest.approx(25.0)
    assert not result.iloc[0]["spatially_interpolated"]


def test_coarse_grid_is_interpolated_and_marked() -> None:
    frame = pd.DataFrame(
        {
            "latitude": [14.0, 14.0, 14.1, 14.1],
            "longitude": [-90.1, -90.0, -90.1, -90.0],
            "temperature_mean_c": [0.0, 10.0, 20.0, 30.0],
        }
    )

    result = harmonize_spatial(frame, value_columns=["temperature_mean_c"])

    center = result.loc[result["latitude"].eq(14.025) & result["longitude"].eq(-90.025)].iloc[0]
    assert center["temperature_mean_c"] == pytest.approx(12.5)
    assert result["spatially_interpolated"].all()


def test_time_aggregation_applies_variable_specific_rules() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-11"]),
            "latitude": [14.5] * 3,
            "longitude": [-90.5] * 3,
            "rainfall_mm": [1.0, 2.0, 4.0],
            "temperature_mean_c": [20.0, 22.0, 24.0],
            "soil_moisture": [0.2, 0.4, 0.6],
            "ndvi": [0.3, 0.8, 0.5],
        }
    )

    result = aggregate_time(
        frame,
        rules={
            "rainfall_mm": "sum",
            "temperature_mean_c": "mean",
            "soil_moisture": "mean",
            "ndvi": "max",
        },
        frequency="dekad",
    )

    first = result.iloc[0]
    assert first["date"] == pd.Timestamp("2024-01-01")
    assert first["rainfall_mm"] == pytest.approx(3.0)
    assert first["temperature_mean_c"] == pytest.approx(21.0)
    assert first["soil_moisture"] == pytest.approx(0.3)
    assert first["ndvi"] == pytest.approx(0.8)
    assert len(result) == 2


def test_monthly_aggregation_uses_month_start_and_keeps_spatial_cells_separate() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-01"]),
            "latitude": [14.5, 14.5, 14.6],
            "longitude": [-90.5, -90.5, -90.6],
            "rainfall_mm": [1.0, 2.0, 4.0],
        }
    )

    result = aggregate_time(frame, rules={"rainfall_mm": "sum"}, frequency="month")

    assert result["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert result["rainfall_mm"].tolist() == pytest.approx([3.0, 4.0])


def test_monthly_anomalies_use_a_separate_normal_per_cell_and_month() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2023-01-01", "2022-02-01"]),
            "latitude": [14.5, 14.5, 14.5],
            "longitude": [-90.5, -90.5, -90.5],
            "rainfall_mm": [10.0, 30.0, 8.0],
        }
    )

    result = add_monthly_anomalies(frame, value_columns=["rainfall_mm"])

    assert result["rainfall_mm_normal"].tolist() == pytest.approx([20.0, 20.0, 8.0])
    assert result["rainfall_mm_anomaly"].tolist() == pytest.approx([-10.0, 10.0, 0.0])


def test_missing_measurements_are_ignored_without_becoming_zero_rainfall() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "latitude": [14.5, 14.5],
            "longitude": [-90.5, -90.5],
            "rainfall_mm": [None, 4.0],
        }
    )

    result = aggregate_time(frame, rules={"rainfall_mm": "sum"}, frequency="month")

    assert result.iloc[0]["rainfall_mm"] == pytest.approx(4.0)


def test_all_missing_rainfall_remains_missing_after_aggregation() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "latitude": [14.5, 14.5],
            "longitude": [-90.5, -90.5],
            "rainfall_mm": [None, None],
        }
    )

    result = aggregate_time(frame, rules={"rainfall_mm": "sum"}, frequency="month")

    assert pd.isna(result.iloc[0]["rainfall_mm"])
