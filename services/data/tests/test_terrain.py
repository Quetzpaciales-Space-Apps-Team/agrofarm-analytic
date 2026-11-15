"""Terrain module: slope, aspect, D8 flow accumulation and provisional risk zones, checked on
small synthetic DEMs with known answers, plus tile mosaicking, nodata and the clean output."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine, from_origin

from agrofarm_data.cli import main
from agrofarm_data.terrain import analysis, zones
from agrofarm_data.terrain.dem import TerrainError, read_dem
from agrofarm_data.terrain.pipeline import DATASET, LAYER_UNITS, write_clean

LAT = 14.6
CELL = 0.0003  # degrees, close to the 1 arc second of GLO-30
LAT_M = 110_700.0  # metres per degree of latitude near 14.6 N (independent of the module)
LON_M = LAT_M * math.cos(math.radians(LAT))


def grid(rows: int, cols: int) -> Affine:
    return from_origin(-90.5, LAT + rows * CELL / 2, CELL, CELL)


def plane(direction: str, angle: float, rows: int = 12, cols: int = 12) -> NDArray[np.float64]:
    """A plane rising toward `direction` at `angle` degrees."""
    row, col = np.mgrid[0:rows, 0:cols].astype("float64")
    east, north = col * CELL * LON_M, -row * CELL * LAT_M  # rows grow southward
    toward = {"east": east, "west": -east, "north": north, "south": -north}[direction]
    heights: NDArray[np.float64] = np.tan(math.radians(angle)) * toward
    return heights


# ---- geometry ----------------------------------------------------------------------------


def test_metres_per_degree_follow_latitude() -> None:
    lon0, lat0 = analysis.metres_per_degree(np.array([0.0]))
    lon60, lat60 = analysis.metres_per_degree(np.array([60.0]))
    assert lat0[0] == pytest.approx(110_574, abs=100)
    assert lon0[0] == pytest.approx(lat0[0])
    assert lon60[0] == pytest.approx(lat60[0] * 0.5, rel=1e-6)  # cos(60 deg) = 0.5


# ---- slope and aspect --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rises_toward", "aspect"), [("east", 270.0), ("west", 90.0), ("north", 180.0), ("south", 0.0)]
)
def test_tilted_plane_has_constant_slope_and_known_aspect(rises_toward: str, aspect: float) -> None:
    dem = plane(rises_toward, 10.0)
    slope, facing = analysis.slope_aspect(dem, grid(*dem.shape))
    assert slope == pytest.approx(10.0, rel=0.01)
    assert np.abs((facing - aspect + 180.0) % 360.0 - 180.0).max() < 1.0  # wraps at north


def test_flat_dem_has_zero_slope_and_null_aspect() -> None:
    dem = np.full((8, 8), 1500.0)
    slope, facing = analysis.slope_aspect(dem, grid(8, 8))
    assert (slope == 0).all()
    assert np.isnan(facing).all()


def test_nodata_cells_and_their_neighbours_have_no_slope() -> None:
    dem = plane("east", 5.0)
    dem[5, 5] = np.nan
    slope, facing = analysis.slope_aspect(dem, grid(*dem.shape))
    assert np.isnan(slope[4:7, 4:7]).all()
    assert np.isnan(facing[4:7, 4:7]).all()
    assert not np.isnan(slope[2, 2])


def test_slope_uses_metres_not_degrees_of_longitude() -> None:
    # The same height step per cell is steeper at the pole side of a tile than at the equator
    # because a degree of longitude shrinks with cos(latitude).
    dem = np.tile(np.arange(10, dtype="float64") * 30.0, (10, 1))  # 30 m rise per cell eastward
    near_equator = from_origin(-90.5, 0.01, CELL, CELL)
    high_latitude = from_origin(-90.5, 60.0, CELL, CELL)
    slope_eq, _ = analysis.slope_aspect(dem, near_equator)
    slope_hi, _ = analysis.slope_aspect(dem, high_latitude)
    assert slope_hi[5, 5] > slope_eq[5, 5] + 15


# ---- depression filling and flow accumulation --------------------------------------------


def test_fill_raises_a_pit_to_its_spill_level() -> None:
    dem = plane("south", 5.0, 9, 9) + 100.0
    dem[4, 4] -= 50.0  # an isolated pit
    filled = analysis.fill_depressions(dem)
    assert (filled >= dem).all()
    assert filled[4, 4] >= dem[3, 4] - 1e-6  # raised to at least the lowest way out
    unchanged = np.ones(dem.shape, dtype=bool)
    unchanged[4, 4] = False
    assert np.allclose(filled[unchanged], dem[unchanged], atol=1e-6)


def test_plane_draining_south_accumulates_down_each_column() -> None:
    dem = plane("north", 5.0, 10, 6)  # high in the north
    acc = analysis.flow_accumulation(dem, grid(*dem.shape))
    for row in range(10):
        assert acc[row, 3] == row + 1  # the cell itself and everything north of it


def test_v_valley_accumulation_grows_along_the_axis() -> None:
    rows, cols = 15, 11
    row, col = np.mgrid[0:rows, 0:cols].astype("float64")
    dem = 20.0 * np.abs(col - 5) + 2.0 * (rows - 1 - row)  # axis at col 5, outlet at the south
    acc = analysis.flow_accumulation(dem, grid(rows, cols))
    axis = acc[:, 5]
    assert (np.diff(axis) > 0).all()
    assert axis[-1] > 0.6 * rows * cols  # most of the grid drains through the outlet
    assert acc[0, 0] == 1  # a ridge top receives nothing


def test_every_cell_is_counted_once_even_with_pits_and_flats() -> None:
    rng = np.random.default_rng(7)
    dem = np.round(rng.normal(100.0, 3.0, (30, 30)), 0)  # rounded: many flats and pits
    dem[10:14, 10:14] = 80.0  # a flat basin
    acc = analysis.flow_accumulation(dem, grid(30, 30))
    filled = analysis.fill_depressions(dem)
    assert np.isfinite(acc).all()
    assert acc.min() >= 1
    assert np.isfinite(filled).all()
    # Cells that pass water on nowhere (outlets) hold the whole catchment: they sum to all cells.
    outlets = analysis.outlet_mask(dem, grid(30, 30))
    assert acc[outlets].sum() == dem.size


def test_flow_accumulation_skips_nodata() -> None:
    dem = plane("north", 5.0, 8, 8)
    dem[0:2, 0:2] = np.nan
    acc = analysis.flow_accumulation(dem, grid(8, 8))
    assert np.isnan(acc[0:2, 0:2]).all()
    assert acc[2:, 2:].min() >= 1


# ---- provisional zones -------------------------------------------------------------------


def test_erosion_zone_classes_follow_slope_and_concentrated_flow() -> None:
    slope = np.array([[2.0, 8.0, 20.0, 8.0, 2.0, np.nan]])
    acc = np.array([[1.0, 1.0, 1.0, 500.0, 500.0, 1.0]])
    zone = zones.erosion_zone(slope, acc, concentrated_flow_min=100.0)
    assert zone.tolist() == [[0, 1, 2, 2, 0, zones.NODATA]]  # flat land stays low even if wet
    assert zone.dtype == np.uint8


def test_waterlogging_zone_needs_low_slope_and_convergence() -> None:
    slope = np.array([[0.5, 0.5, 0.5, 6.0, np.nan]])
    acc = np.array([[1.0, 300.0, 5000.0, 5000.0, 5000.0]])
    zone = zones.waterlogging_zone(slope, acc, moderate_min=200.0, high_min=1000.0)
    assert zone.tolist() == [[0, 1, 2, 0, zones.NODATA]]


def test_percentile_limits_never_fall_below_the_absolute_floor() -> None:
    slope = np.full((10, 10), 1.0)
    acc = np.ones((10, 10))  # nothing converges: every percentile is 1 cell
    acc[0, 0] = 5000.0
    erosion, waterlogging, limits = zones.classify(slope, acc)
    floor = zones.MIN_CONVERGENCE_CELLS
    assert limits["concentrated_flow_min_cells"] == floor
    assert limits["waterlogging_moderate_min_cells"] == floor
    assert (waterlogging == zones.LOW).sum() == 99  # only the one real convergent cell
    assert waterlogging[0, 0] == zones.HIGH
    assert (erosion == zones.LOW).all()  # slope 1 degree: not an erosion zone


def test_thresholds_are_documented_constants() -> None:
    assert zones.EROSION_SLOPE_MODERATE_DEG < zones.EROSION_SLOPE_HIGH_DEG
    assert 0 < zones.CONVERGENCE_PERCENTILE < 100
    assert zones.WATERLOGGING_MAX_SLOPE_DEG < zones.EROSION_SLOPE_MODERATE_DEG
    assert "provisional" in zones.PROVENANCE.lower()
    assert "agronomist" in zones.PROVENANCE.lower()


# ---- reading and mosaicking tiles --------------------------------------------------------

TILE_CELL = 0.01
TILE_PX = 20


def write_tile(path: Path, west: float, north: float, nodata: float | None = None) -> None:
    """A 20 x 20 tile whose height rises 10 degrees toward the east, continuous across tiles."""
    col = np.arange(TILE_PX, dtype="float64")
    longitude = west + (col + 0.5) * TILE_CELL
    height = (
        np.tan(math.radians(10.0))
        * (longitude + 91.0)
        * 111_000.0
        * math.cos(math.radians(north - 0.1))
    )
    data = np.tile(height, (TILE_PX, 1)).astype("float32")
    if nodata is not None:
        data[:, 0] = nodata
    profile = {"driver": "GTiff", "height": TILE_PX, "width": TILE_PX, "count": 1,
               "dtype": "float32", "crs": "EPSG:4326", "nodata": nodata,
               "transform": from_origin(west, north, TILE_CELL, TILE_CELL)}  # fmt: skip
    with rasterio.open(path, "w", **profile) as out:
        out.write(data, 1)


@pytest.fixture
def tiles(tmp_path: Path) -> Path:
    folder = tmp_path / "dem"
    folder.mkdir()
    write_tile(folder / "tile_a.tif", -90.4, 14.7)  # lon -90.4..-90.2
    write_tile(folder / "tile_b.tif", -90.2, 14.7)  # lon -90.2..-90.0, shares the edge
    return folder


BOX = (-90.3, 14.55, -90.1, 14.65)  # crosses the edge between the two tiles


def test_box_across_a_tile_edge_is_mosaicked_without_a_seam(tiles: Path) -> None:
    dem = read_dem(tiles, BOX)
    assert {path.name for path in dem.tiles} == {"tile_a.tif", "tile_b.tif"}
    assert not np.isnan(dem.values).any()
    slope, _ = analysis.slope_aspect(dem.values, dem.transform)
    seam = slope[:, dem.values.shape[1] // 2 - 2 : dem.values.shape[1] // 2 + 2]
    assert seam == pytest.approx(10.0, rel=0.05)


def test_only_tiles_that_touch_the_box_are_used(tiles: Path) -> None:
    dem = read_dem(tiles, (-90.35, 14.55, -90.25, 14.65))
    assert [path.name for path in dem.tiles] == ["tile_a.tif"]


def test_declared_nodata_becomes_nan(tmp_path: Path) -> None:
    folder = tmp_path / "dem"
    folder.mkdir()
    write_tile(folder / "t.tif", -90.4, 14.7, nodata=-32767.0)
    dem = read_dem(folder, (-90.4, 14.5, -90.2, 14.7), margin_deg=0.0)
    assert np.isnan(dem.values[:, 0]).all()
    assert not np.isnan(dem.values[:, 1:]).any()


def test_box_without_tiles_is_rejected(tiles: Path) -> None:
    with pytest.raises(TerrainError, match="No DEM tile"):
        read_dem(tiles, (10.0, 10.0, 10.5, 10.5))


def test_empty_folder_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TerrainError, match="No DEM tile"):
        read_dem(tmp_path, BOX)


def test_tiles_on_different_grids_are_rejected(tiles: Path) -> None:
    write_tile(tiles / "tile_c.tif", -90.3 + 0.004, 14.7)  # shifted by 0.4 cell
    with pytest.raises(TerrainError, match="grid"):
        read_dem(tiles, BOX)


def test_window_is_cropped_to_the_box_after_the_margin(tiles: Path) -> None:
    dem = read_dem(tiles, BOX, margin_deg=0.03)
    assert dem.crop.width == 20
    assert dem.crop.height == 10
    assert dem.values.shape[1] > dem.crop.width  # the margin is wider than the box


# ---- clean output ------------------------------------------------------------------------


def test_clean_output_has_table_layers_and_an_honest_manifest(tiles: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"
    manifest_path = write_clean(tiles, out, BOX, area_name="test area")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    table = pd.read_parquet(out / manifest["file"])

    assert manifest["dataset"] == DATASET
    assert manifest["bbox"] == {"west": -90.3, "south": 14.55, "east": -90.1, "north": 14.65}
    assert "Copernicus" in manifest["source"]
    assert "attribution" in manifest["license"].lower()
    assert manifest["license_verified"] is False
    assert {tile["file"] for tile in manifest["tiles"]} == {"tile_a.tif", "tile_b.tif"}
    digest = hashlib.sha256((tiles / "tile_a.tif").read_bytes()).hexdigest()
    assert {tile["sha256"] for tile in manifest["tiles"]} >= {digest}
    assert manifest["units"] == LAYER_UNITS
    assert manifest["units"]["slope"] == "degrees"
    assert manifest["thresholds"]["erosion_slope_high_deg"] == zones.EROSION_SLOPE_HIGH_DEG
    assert "provisional" in manifest["thresholds"]["status"].lower()
    assert manifest["depressions"].startswith("filled")
    joined = " ".join(manifest["limitations"]).lower()
    assert "surface model" in joined and "buildings" in joined and "heuristic" in joined

    assert set(LAYER_UNITS) == {
        "elevation", "slope", "aspect", "flow_accumulation", "erosion_risk", "waterlogging_risk",
    }  # fmt: skip
    assert {layer["file"] for layer in manifest["layers"]} == {
        f"layers/{name}.tif" for name in LAYER_UNITS
    }
    with rasterio.open(out / "layers" / "slope.tif") as raster:
        assert raster.profile["tiled"]
        assert raster.shape == (10, 20)
        assert np.nanmean(raster.read(1)) == pytest.approx(10.0, rel=0.05)
    with rasterio.open(out / "layers" / "erosion_risk.tif") as raster:
        assert raster.dtypes[0] == "uint8"
        assert raster.nodata == zones.NODATA

    shares = table[table["kind"] == "zone"].groupby("layer")["area_share"].sum()
    assert shares.to_dict() == pytest.approx({"erosion_risk": 1.0, "waterlogging_risk": 1.0})
    slope_row = table[table["layer"] == "slope"].iloc[0]
    assert slope_row["mean"] == pytest.approx(10.0, rel=0.05)
    assert slope_row["unit"] == "degrees"


def test_rerunning_gives_identical_output(tiles: Path, tmp_path: Path) -> None:
    first = write_clean(tiles, tmp_path / "a", BOX, "x")
    second = write_clean(tiles, tmp_path / "b", BOX, "x")
    one, two = (json.loads(path.read_text("utf-8")) for path in (first, second))
    assert one["layers"] == two["layers"]
    assert one["checksum_sha256"] == two["checksum_sha256"]


def test_command_line_writes_the_terrain_dataset(
    tiles: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clean"
    bbox = [str(value) for value in BOX]
    args = ["terrain", "--input", str(tiles), "--out", str(out), "--bbox", *bbox]
    assert main([*args, "--area-name", "test area"]) == 0
    assert (out / f"{DATASET}.manifest.json").is_file()
    assert "layers" in capsys.readouterr().out
