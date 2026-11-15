"""Erosion-risk and waterlogging-risk zones: PROVISIONAL HEURISTICS, not agronomic truth.

Both zones are simple rules on slope and flow accumulation from a 30 m surface model. They
flag where terrain alone makes a problem more likely. They do not know soil, cover, rainfall,
drainage works or management. Every number below is a team assumption that an agronomist must
review before a zone is shown to a farmer as anything more than a "look here" hint.

Zone codes (uint8): 0 low, 1 moderate, 2 high, 255 no data.
"""

import numpy as np
from numpy.typing import NDArray

PROVENANCE = (
    "Provisional heuristics. The class limits are team assumptions loosely based on common "
    "slope classes (gentle under 5 degrees, steep above about 15 degrees); they are not "
    "calibrated to Guatemala and are to be reviewed by an agronomist."
)

NODATA = 255
LOW, MODERATE, HIGH = 0, 1, 2

# Erosion: slope class first, then concentrated flow raises the class by one.
EROSION_SLOPE_MODERATE_DEG = 5.0  # team assumption: below this, water erosion is usually minor
EROSION_SLOPE_HIGH_DEG = 15.0  # team assumption: above this, rill and gully risk is high
# A cell is "concentrated flow" when its accumulation is at or above this percentile of the
# cells of the area. It is relative to the area, so it moves with the box (team assumption).
CONVERGENCE_PERCENTILE = 95.0

# A percentile alone is meaningless on smooth ground where most cells receive almost nothing
# (the 75th percentile can be 6 cells), so every accumulation limit is at least this many cells
# (100 cells is about 9 hectares of catchment). Team assumption.
MIN_CONVERGENCE_CELLS = 100.0

# Waterlogging: low slope AND water converging from upslope.
WATERLOGGING_MAX_SLOPE_DEG = 2.0  # team assumption: gentle enough for water to stand
WATERLOGGING_MODERATE_PERCENTILE = 90.0  # accumulation percentile of the area (assumption)
WATERLOGGING_HIGH_PERCENTILE = 98.0  # accumulation percentile of the area (assumption)

Float = NDArray[np.float64]
Zone = NDArray[np.uint8]


def percentile_of_valid(values: Float, percentile: float) -> float:
    """Percentile over the cells that have data, never below MIN_CONVERGENCE_CELLS."""
    valid = values[~np.isnan(values)]
    if not valid.size:
        return float("nan")
    return max(float(np.percentile(valid, percentile)), MIN_CONVERGENCE_CELLS)


def erosion_zone(slope: Float, accumulation: Float, concentrated_flow_min: float) -> Zone:
    """0 low under 5 degrees, 1 moderate 5 to 15, 2 high above 15; +1 (max 2) on convergence.

    Concentrated flow only raises the class on slopes of at least the moderate limit: a
    convergent flat is a waterlogging question, not an erosion one.
    """
    zone = np.full(slope.shape, LOW, dtype=np.uint8)
    zone[slope >= EROSION_SLOPE_MODERATE_DEG] = MODERATE
    zone[slope >= EROSION_SLOPE_HIGH_DEG] = HIGH
    with np.errstate(invalid="ignore"):
        converging = (accumulation >= concentrated_flow_min) & (slope >= EROSION_SLOPE_MODERATE_DEG)
    zone[converging] = np.minimum(zone[converging] + 1, HIGH).astype(np.uint8)
    zone[np.isnan(slope) | np.isnan(accumulation)] = NODATA
    return zone


def waterlogging_zone(
    slope: Float, accumulation: Float, moderate_min: float, high_min: float
) -> Zone:
    """Low slope (under 2 degrees) plus accumulation: moderate at the 90th percentile of the
    area, high at the 98th (both with the absolute floor). Anything steeper is low."""
    zone = np.full(slope.shape, LOW, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        flat = slope < WATERLOGGING_MAX_SLOPE_DEG
        zone[flat & (accumulation >= moderate_min)] = MODERATE
        zone[flat & (accumulation >= high_min)] = HIGH
    zone[np.isnan(slope) | np.isnan(accumulation)] = NODATA
    return zone


def classify(slope: Float, accumulation: Float) -> tuple[Zone, Zone, dict[str, float]]:
    """Both zones for an area, with the accumulation limits that were derived from it."""
    limits = {
        "concentrated_flow_min_cells": percentile_of_valid(accumulation, CONVERGENCE_PERCENTILE),
        "waterlogging_moderate_min_cells": percentile_of_valid(
            accumulation, WATERLOGGING_MODERATE_PERCENTILE
        ),
        "waterlogging_high_min_cells": percentile_of_valid(
            accumulation, WATERLOGGING_HIGH_PERCENTILE
        ),
    }
    erosion = erosion_zone(slope, accumulation, limits["concentrated_flow_min_cells"])
    waterlogging = waterlogging_zone(
        slope,
        accumulation,
        limits["waterlogging_moderate_min_cells"],
        limits["waterlogging_high_min_cells"],
    )
    return erosion, waterlogging, limits
