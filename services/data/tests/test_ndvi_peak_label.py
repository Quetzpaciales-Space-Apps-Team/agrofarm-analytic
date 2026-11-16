"""Seasonal peak NDVI on cropland per grid cell: the proxy label of the experimental model."""

import math

import numpy as np
import pytest

from agrofarm_data.labels.ndvi_peak import (
    EARTH_RADIUS_M,
    LabelError,
    cropland_weighted_cells,
    sample_sinusoidal,
    seasonal_peak,
    sinusoidal_xy,
)

NAN = float("nan")


def test_sinusoidal_projection_shrinks_longitude_with_latitude() -> None:
    x_equator, y_equator = sinusoidal_xy(np.array([0.0]), np.array([-90.0]))
    x_north, y_north = sinusoidal_xy(np.array([60.0]), np.array([-90.0]))

    assert y_equator[0] == pytest.approx(0.0)
    assert x_equator[0] == pytest.approx(-EARTH_RADIUS_M * math.pi / 2)
    assert x_north[0] == pytest.approx(x_equator[0] * 0.5)
    assert y_north[0] == pytest.approx(EARTH_RADIUS_M * math.pi / 3)


def test_sampling_scales_the_values_and_never_turns_fill_or_out_of_range_into_ndvi() -> None:
    # Two by two raster, 1000 m pixels, top-left corner at x=0, y=2000.
    raw = np.array([[8000, -3000], [12000, -2500]], dtype="int16")
    transform = (1000.0, 0.0, 0.0, 0.0, -1000.0, 2000.0)
    x = np.array([500.0, 1500.0, 500.0, 1500.0, 5000.0])
    y = np.array([1500.0, 1500.0, 500.0, 500.0, 500.0])

    ndvi = sample_sinusoidal(raw, transform, x, y)

    assert ndvi[0] == pytest.approx(0.8)
    assert np.isnan(ndvi[1])  # fill value
    assert np.isnan(ndvi[2])  # above the valid range
    assert np.isnan(ndvi[3])  # below the valid range
    assert np.isnan(ndvi[4])  # outside the raster


def test_seasonal_peak_is_the_maximum_and_needs_enough_composites() -> None:
    stack = np.array(
        [
            [[0.2, 0.5], [NAN, NAN]],
            [[0.7, NAN], [NAN, 0.1]],
            [[0.4, 0.6], [NAN, NAN]],
        ]
    )

    peak = seasonal_peak(stack, min_composites=2)

    assert peak[0, 0] == pytest.approx(0.7)
    assert peak[0, 1] == pytest.approx(0.6)
    assert np.isnan(peak[1, 0])  # never observed
    assert np.isnan(peak[1, 1])  # one composite is not a season


def test_cells_are_weighted_by_cropland_fraction() -> None:
    peak = np.array([[0.8, 0.4], [0.6, 0.2]])
    cropland = np.array([[0.5, 0.0], [0.5, 0.0]])

    cells, weight = cropland_weighted_cells(peak, cropland, factor=2, min_cover=0.8)

    assert cells.shape == (1, 1)
    assert cells[0, 0] == pytest.approx(0.7)  # the two sub-cells without cropland do not count
    assert weight[0, 0] == pytest.approx(0.25)  # mean cropland fraction of the cell


def test_a_cell_with_too_little_observed_cropland_has_no_label() -> None:
    peak = np.array([[0.8, NAN], [NAN, NAN]])
    cropland = np.array([[0.2, 0.3], [0.3, 0.2]])

    cells, _ = cropland_weighted_cells(peak, cropland, factor=2, min_cover=0.8)

    assert np.isnan(cells[0, 0])


def test_a_cell_without_cropland_has_no_label_and_missing_cropland_counts_as_none() -> None:
    peak = np.array([[0.8, 0.7], [0.6, 0.5]])
    cropland = np.array([[0.0, NAN], [0.0, 0.0]])

    cells, weight = cropland_weighted_cells(peak, cropland, factor=2, min_cover=0.8)

    assert np.isnan(cells[0, 0])
    assert weight[0, 0] == pytest.approx(0.0)


def test_grids_that_do_not_divide_into_cells_are_rejected() -> None:
    with pytest.raises(LabelError):
        cropland_weighted_cells(np.zeros((3, 4)), np.zeros((3, 4)), factor=2, min_cover=0.8)
    with pytest.raises(LabelError):
        cropland_weighted_cells(np.zeros((2, 2)), np.zeros((4, 4)), factor=2, min_cover=0.8)
