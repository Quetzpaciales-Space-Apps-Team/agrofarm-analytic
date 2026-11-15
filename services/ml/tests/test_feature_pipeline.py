"""ML-010: the feature pipeline. Label independent (no yield, no NDVI target) and offline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrofarm_ml.features import ENSO_LAGS, validation
from agrofarm_ml.features.anomalies import add_anomalies, climatology
from agrofarm_ml.features.clean_tables import CleanTableError, read_clean_table
from agrofarm_ml.features.pipeline import build_cell_period_table

GOLDEN = Path(__file__).parent / "fixtures" / "features_golden"
VALUES = ("rainfall_mm", "temperature_mean_c")


def expand_to_days(monthly: pd.DataFrame) -> pd.DataFrame:
    """The golden file lists monthly values. Spread them over the days of the month so the
    monthly rainfall total is that value and the mean temperature is that value."""
    rows = []
    for item in monthly.itertuples():
        days = pd.date_range(item.date.to_period("M").to_timestamp(), periods=item.date.days_in_month)
        rows.append(
            pd.DataFrame(
                {
                    "date": days,
                    "cell_id": item.cell_id,
                    "rainfall_mm": item.rainfall_mm / len(days),
                    "temperature_mean_c": item.temperature_mean_c,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def climate() -> pd.DataFrame:
    return expand_to_days(pd.read_csv(GOLDEN / "climate.csv", parse_dates=["date"]))


@pytest.fixture
def enso() -> pd.DataFrame:
    return pd.read_csv(GOLDEN / "enso.csv", parse_dates=["month"])


def test_golden_table_one_row_per_cell_and_month_with_anomalies_and_lags(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    table = build_cell_period_table(climate, enso, value_columns=VALUES)
    table = table.set_index(["cell_id", "month"])
    assert len(table) == 9  # A has 6 months, B has 3
    # A, January: 10, 30, 50 mm, climatology 30 mm; 2000 anomaly is -20 mm
    assert table.loc[("A", pd.Timestamp("2000-01-01")), "rainfall_mm"] == pytest.approx(10.0)
    assert table.loc[("A", pd.Timestamp("2000-01-01")), "rainfall_mm_anomaly"] == pytest.approx(-20)
    assert table.loc[("A", pd.Timestamp("2002-01-01")), "rainfall_mm_anomaly"] == pytest.approx(20)
    # A, February: 20, 40 and one missing year. The mean uses the two real values (30).
    assert table.loc[("A", pd.Timestamp("2001-02-01")), "rainfall_mm_anomaly"] == pytest.approx(10)
    # B has its own climatology: January 1, 3, 5 mm -> 2001 anomaly 0
    assert table.loc[("B", pd.Timestamp("2001-01-01")), "rainfall_mm_anomaly"] == pytest.approx(0)
    assert table.loc[("B", pd.Timestamp("2001-01-01")), "temperature_mean_c_anomaly"] == 0
    # ENSO lags 1 to 6 look at earlier months. roni at month i is i / 10 (1999-07 is 0).
    row = table.loc[("A", pd.Timestamp("2000-01-01"))]
    assert row["roni_lag1"] == pytest.approx(0.5)  # 1999-12
    assert row["roni_lag6"] == pytest.approx(0.0)  # 1999-07
    assert row["roni_lag0"] == pytest.approx(0.6)  # 2000-01


def test_lags_one_to_six_are_features() -> None:
    assert set(range(1, 7)) <= set(ENSO_LAGS)


def test_missing_values_stay_nan_and_the_row_is_kept(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    table = build_cell_period_table(climate, enso, value_columns=VALUES)
    row = table[(table["cell_id"] == "A") & (table["month"] == "2002-02-01")].iloc[0]
    assert np.isnan(row["rainfall_mm"]) and np.isnan(row["rainfall_mm_anomaly"])
    assert row["temperature_mean_c"] == 25.0  # other columns are not lost


def test_sentinel_values_are_rejected_and_never_reach_the_table(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    for sentinel in (-99.9, -999.0):
        bad = climate.copy()
        bad.loc[0, "rainfall_mm"] = sentinel
        with pytest.raises(validation.InputError, match="sentinel"):
            build_cell_period_table(bad, enso, value_columns=VALUES)
    table = build_cell_period_table(climate, enso, value_columns=VALUES)
    numbers = table.select_dtypes("number").to_numpy()
    assert not (numbers == -99.9).any()
    assert not (numbers == -999.0).any()


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("rainfall_mm", -5.0, "outside"),
        ("temperature_mean_c", 80.0, "outside"),
        ("date", "not a date", "date"),
    ],
)
def test_climate_validation_names_the_column(
    climate: pd.DataFrame, enso: pd.DataFrame, column: str, value: object, match: str
) -> None:
    bad = climate.astype({"date": "object"}) if column == "date" else climate.copy()
    bad.loc[0, column] = value
    with pytest.raises(validation.InputError, match=match):
        build_cell_period_table(bad, enso, value_columns=VALUES)


def test_validation_rejects_missing_columns_duplicates_and_empty_tables(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    with pytest.raises(validation.InputError, match="rainfall_mm"):
        validation.validate_climate(climate.drop(columns="rainfall_mm"))
    with pytest.raises(validation.InputError, match="duplicate"):
        validation.validate_climate(pd.concat([climate, climate.head(1)]))
    with pytest.raises(validation.InputError, match="no rows"):
        validation.validate_climate(climate.head(0))
    with pytest.raises(validation.InputError, match="roni"):
        validation.validate_enso(enso.drop(columns="roni"))
    with pytest.raises(validation.InputError, match="duplicate"):
        validation.validate_enso(pd.concat([enso, enso.head(1)]))
    broken = enso.copy()
    broken.loc[0, "roni"] = 9.0
    with pytest.raises(validation.InputError, match="outside"):
        validation.validate_enso(broken)


def test_a_climate_table_without_cell_id_is_one_station(enso: pd.DataFrame) -> None:
    daily = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", "2000-01-31"), "rainfall_mm": 1.0}
    )
    table = build_cell_period_table(daily, enso)
    assert set(table["cell_id"]) == {"station"}
    assert table["rainfall_mm"].iloc[0] == pytest.approx(31.0)


def test_climatology_can_use_only_reference_years_to_avoid_leakage(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    table = build_cell_period_table(climate, enso, value_columns=VALUES, reference_years=[2000])
    january = table[(table["cell_id"] == "A") & (table["month"].dt.month == 1)]
    assert january.set_index("month")["rainfall_mm_anomaly"].tolist() == [0.0, 20.0, 40.0]


def test_anomaly_without_any_reference_value_is_nan() -> None:
    table = pd.DataFrame(
        {
            "cell_id": ["A", "A"],
            "month": pd.to_datetime(["2000-01-01", "2001-02-01"]),
            "year": [2000, 2001],
            "month_of_year": [1, 2],
            "x": [1.0, np.nan],
        }
    )
    out = add_anomalies(table, ["x"], reference_years=[2000])
    assert out["x_anomaly"].iloc[0] == 0.0
    assert np.isnan(out["x_anomaly"].iloc[1])
    assert climatology(table, ["x"]).shape[0] == 1


def test_cropland_fraction_is_a_weight_never_a_mask(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    static = pd.DataFrame({"cell_id": ["A", "B"], "cropland_fraction": [0.8, 0.0]})
    table = build_cell_period_table(climate, enso, value_columns=VALUES, static=static)
    assert set(table["cell_id"]) == {"A", "B"}, "a cell with no cropland is kept"
    assert table.loc[table["cell_id"] == "B", "cropland_fraction"].eq(0.0).all()
    assert table.loc[table["cell_id"] == "A", "cropland_fraction"].eq(0.8).all()


def test_static_table_must_have_one_row_per_cell_and_a_valid_fraction(
    climate: pd.DataFrame, enso: pd.DataFrame
) -> None:
    twice = pd.DataFrame({"cell_id": ["A", "A"], "cropland_fraction": [0.1, 0.2]})
    with pytest.raises(validation.InputError, match="duplicate"):
        build_cell_period_table(climate, enso, value_columns=VALUES, static=twice)
    odd = pd.DataFrame({"cell_id": ["A"], "cropland_fraction": [1.5]})
    with pytest.raises(validation.InputError, match="outside"):
        build_cell_period_table(climate, enso, value_columns=VALUES, static=odd)


def write_clean(directory: Path, frame: pd.DataFrame, **overrides: object) -> None:
    table = directory / "demo.parquet"
    frame.to_parquet(table, index=False)
    manifest = {
        "dataset": "demo",
        "file": "demo.parquet",
        "checksum_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "source_version": "test",
    } | overrides
    (directory / "demo.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_clean_tables_are_read_by_manifest_and_checked(tmp_path: Path, enso: pd.DataFrame) -> None:
    write_clean(tmp_path, enso)
    frame, manifest = read_clean_table(tmp_path, "demo")
    assert len(frame) == len(enso)
    assert manifest["dataset"] == "demo"


def test_clean_table_with_a_wrong_checksum_or_missing_files_is_refused(
    tmp_path: Path, enso: pd.DataFrame
) -> None:
    write_clean(tmp_path, enso, checksum_sha256="0" * 64)
    with pytest.raises(CleanTableError, match="checksum"):
        read_clean_table(tmp_path, "demo")
    with pytest.raises(CleanTableError, match="manifest"):
        read_clean_table(tmp_path, "other")
    write_clean(tmp_path, enso, file="gone.parquet")
    with pytest.raises(CleanTableError, match="gone.parquet"):
        read_clean_table(tmp_path, "demo")


def test_clean_table_manifest_may_not_point_outside_its_folder(
    tmp_path: Path, enso: pd.DataFrame
) -> None:
    write_clean(tmp_path, enso, file="../escape.parquet")
    with pytest.raises(CleanTableError, match="outside"):
        read_clean_table(tmp_path, "demo")


def test_the_pipeline_does_not_modify_its_inputs(climate: pd.DataFrame, enso: pd.DataFrame) -> None:
    before = climate.copy(), enso.copy()
    build_cell_period_table(climate, enso, value_columns=VALUES)
    pd.testing.assert_frame_equal(climate, before[0])
    pd.testing.assert_frame_equal(enso, before[1])
