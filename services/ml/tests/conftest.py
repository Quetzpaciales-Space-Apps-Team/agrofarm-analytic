"""Tiny synthetic datasets with a known ENSO effect on rainfall. No real data, no network."""

import numpy as np
import pandas as pd
import pytest

ENSO_EFFECT = 0.35  # one degree of RONI removes 35% of the rain in the synthetic world
SEASONAL_MM_PER_DAY = [0.1, 0.1, 0.3, 1.0, 5.0, 9.0, 7.0, 7.0, 9.0, 5.0, 1.0, 0.2]


def make_enso(start: str = "1989-01-01", end: str = "2020-12-01") -> pd.DataFrame:
    months = pd.date_range(start, end, freq="MS")
    step = np.arange(len(months))
    roni = 1.5 * np.sin(2 * np.pi * step / 44.0)
    return pd.DataFrame({"month": months, "oni": roni, "roni": roni})


def make_climate(enso: pd.DataFrame, start: str = "1990-01-01", end: str = "2020-12-31",
                 seed: int = 7) -> pd.DataFrame:  # fmt: skip
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    roni = enso.set_index("month")["roni"].reindex(days.to_period("M").to_timestamp()).to_numpy()
    base = np.array([SEASONAL_MM_PER_DAY[month - 1] for month in days.month])
    rain = base * (1 - ENSO_EFFECT * roni) * rng.gamma(shape=4.0, scale=0.25, size=len(days))
    return pd.DataFrame({"date": days, "rainfall_mm": np.clip(rain, 0, None)})


@pytest.fixture(scope="session")
def enso() -> pd.DataFrame:
    return make_enso()


@pytest.fixture(scope="session")
def climate(enso: pd.DataFrame) -> pd.DataFrame:
    return make_climate(enso)
