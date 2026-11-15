"""WorldCover connector: 10 m land cover classes counted (never resampled) into 0.05 degree
cells. Synthetic tiles use 0.005 degree pixels, so one cell is 10 x 10 pixels = 100 pixels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import from_origin

from agrofarm_data.cli import main
from agrofarm_data.connectors.worldcover import (
    CELL_DEGREES,
    CLASSES,
    CROPLAND_CLASS,
    CROPLAND_THRESHOLD,
    MIN_VALID_FRACTION,
    LOW_CROPLAND_FRACTION,
    WorldCoverError,
    aggregate,
    check_declared_cropland,
    polygon_cover,
    write_clean,
)

PIXEL = 0.005
SIDE = 10  # pixels per cell side
GRASS = 30


def blank(rows: int = 100, cols: int = 100, value: int = GRASS) -> NDArray[np.uint8]:
    return np.full((rows, cols), value, dtype="uint8")


def paint(data: NDArray[np.uint8], cell_row: int, cell_col: int, counts: dict[int, int]) -> None:
    """Fill one 10 x 10 cell with the given number of pixels per class, in raster order."""
    flat = np.concatenate([np.full(n, code, dtype="uint8") for code, n in counts.items()])
    assert flat.size == SIDE * SIDE
    r, c = cell_row * SIDE, cell_col * SIDE
    data[r : r + SIDE, c : c + SIDE] = flat.reshape(SIDE, SIDE)


def write_tile(
    path: Path,
    data: NDArray[np.uint8],
    west: float,
    north: float,
    nodata: int | None = 0,
    block: int | None = None,
    pixel: float = PIXEL,
) -> None:
    profile: dict[str, object] = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
        "dtype": "uint8", "crs": "EPSG:4326", "transform": from_origin(west, north, pixel, pixel),
        "nodata": nodata,
    }  # fmt: skip
    if block:
        profile.update(tiled=True, blockxsize=block, blockysize=block)
    with rasterio.open(path, "w", **profile) as out:
        out.write(data, 1)


def cell(table: pd.DataFrame, lon: float, lat: float) -> pd.Series:
    rows = table[(table["lon"].round(6) == lon) & (table["lat"].round(6) == lat)]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.fixture
def tiles(tmp_path: Path) -> Path:
    """One 100 x 100 tile covering lon -91..-90.5 and lat 14.5..15 (10 x 10 cells)."""
    data = blank()
    paint(data, 0, 0, {40: 30, 10: 20, GRASS: 50})  # cell lon -90.975, lat 14.975
    paint(data, 0, 1, {40: 40, 10: 40, 0: 20})  # 20 pixels without data
    paint(data, 0, 2, {0: 100})  # no data at all
    paint(data, 0, 3, {40: 49, GRASS: 51})  # just below the threshold
    paint(data, 0, 4, {40: 50, GRASS: 50})  # exactly the threshold
    folder = tmp_path / "tiles"
    folder.mkdir()
    write_tile(folder / "ESA_WorldCover_10m_2021_v200_N14W091_Map.tif", data, -91.0, 15.0)
    return folder


BOX = (-91.0, 14.5, -90.5, 15.0)


def test_fractions_are_pixel_counts_over_valid_pixels(tiles: Path) -> None:
    row = cell(aggregate(tiles, BOX), -90.975, 14.975)
    assert row["fraction_cropland"] == pytest.approx(0.30)
    assert row["fraction_tree_cover"] == pytest.approx(0.20)
    assert row["fraction_grassland"] == pytest.approx(0.50)
    assert row["fraction_water"] == 0.0
    assert row["valid_fraction"] == 1.0
    fractions = [row[f"fraction_{name}"] for name in CLASSES.values()]
    assert sum(fractions) == pytest.approx(1.0)


def test_no_data_pixels_leave_the_denominator(tiles: Path) -> None:
    row = cell(aggregate(tiles, BOX), -90.925, 14.975)
    assert row["fraction_cropland"] == pytest.approx(0.5)  # 40 of 80, not 40 of 100
    assert row["valid_fraction"] == pytest.approx(0.8)


def test_cell_without_valid_pixels_is_null_not_zero(tiles: Path) -> None:
    row = cell(aggregate(tiles, BOX), -90.875, 14.975)
    assert pd.isna(row["fraction_cropland"])
    assert pd.isna(row["fraction_tree_cover"])
    assert row["valid_fraction"] == 0.0
    assert not row["cropland_mask"]


def test_threshold_is_documented_and_inclusive(tiles: Path) -> None:
    assert CROPLAND_CLASS == 40
    assert CROPLAND_THRESHOLD == 0.5
    table = aggregate(tiles, BOX)
    assert not cell(table, -90.825, 14.975)["cropland_mask"]  # 0.49
    assert cell(table, -90.775, 14.975)["cropland_mask"]  # 0.50
    assert cell(table, -90.925, 14.975)["cropland_mask"]  # 0.50 of the valid pixels
    assert table["cropland_mask"].dtype == bool


def test_mostly_empty_cell_is_not_in_the_mask(tmp_path: Path) -> None:
    data = blank()
    paint(data, 0, 0, {40: 10, 0: 90})  # all valid pixels are cropland but only 10 % are valid
    write_tile(tmp_path / "t_Map.tif", data, -91.0, 15.0)
    row = cell(aggregate(tmp_path, BOX), -90.975, 14.975)
    assert row["fraction_cropland"] == 1.0
    assert row["valid_fraction"] < MIN_VALID_FRACTION
    assert not row["cropland_mask"]


def test_grid_is_aligned_to_multiples_of_005_and_covers_the_box(tiles: Path) -> None:
    table = aggregate(tiles, (-90.97, 14.52, -90.88, 14.58))  # not on the grid
    lons = sorted(table["lon"].round(6).unique())
    lats = sorted(table["lat"].round(6).unique())
    assert CELL_DEGREES == 0.05
    assert lons == [-90.975, -90.925, -90.875]  # cells -91.0..-90.85
    assert lats == [14.525, 14.575]  # cells 14.5..14.6
    assert len(table) == 6
    assert (((table["lon"] - CELL_DEGREES / 2) / CELL_DEGREES).round(6) % 1 == 0).all()


def test_box_crossing_two_tiles_is_mosaicked(tmp_path: Path) -> None:
    west = blank(value=40)  # lon -91..-90.5, all cropland
    east = blank(value=10)  # lon -90.5..-90, all tree cover
    paint(east, 0, 0, {40: 25, 10: 75})
    write_tile(tmp_path / "ESA_WorldCover_10m_2021_v200_N14W091_Map.tif", west, -91.0, 15.0)
    write_tile(tmp_path / "ESA_WorldCover_10m_2021_v200_N14W090_Map.tif", east, -90.5, 15.0)
    table = aggregate(tmp_path, (-90.6, 14.9, -90.4, 15.0))
    assert len(table) == 8  # four columns (two per tile) and two rows
    assert cell(table, -90.575, 14.975)["fraction_cropland"] == 1.0
    assert cell(table, -90.525, 14.975)["fraction_cropland"] == 1.0
    assert cell(table, -90.475, 14.975)["fraction_cropland"] == pytest.approx(0.25)
    assert cell(table, -90.425, 14.975)["fraction_cropland"] == 0.0
    assert (table["valid_fraction"] == 1.0).all()


def test_box_crossing_two_tiles_vertically(tmp_path: Path) -> None:
    upper = blank(value=40)  # lat 15..15.5
    lower = blank(value=20)  # lat 14.5..15
    write_tile(tmp_path / "upper_Map.tif", upper, -91.0, 15.5)
    write_tile(tmp_path / "lower_Map.tif", lower, -91.0, 15.0)
    table = aggregate(tmp_path, (-91.0, 14.95, -90.95, 15.05))
    assert cell(table, -90.975, 15.025)["fraction_cropland"] == 1.0
    assert cell(table, -90.975, 14.975)["fraction_shrubland"] == 1.0


def test_area_missing_from_the_tiles_is_null_with_zero_valid_fraction(tmp_path: Path) -> None:
    write_tile(tmp_path / "t_Map.tif", blank(value=40), -91.0, 15.0)
    table = aggregate(tmp_path, (-91.0, 14.9, -90.9, 15.1))  # lat 15..15.1 has no tile
    gap = cell(table, -90.975, 15.025)
    assert pd.isna(gap["fraction_cropland"])
    assert gap["valid_fraction"] == 0.0
    assert not gap["cropland_mask"]
    assert cell(table, -90.975, 14.975)["valid_fraction"] == 1.0


def test_blockwise_reading_gives_the_same_counts(tiles: Path, tmp_path: Path) -> None:
    expected = aggregate(tiles, BOX)
    folder = tmp_path / "blocks"
    folder.mkdir()
    with rasterio.open(next(tiles.glob("*.tif"))) as source:
        data = source.read(1)
    # 16 pixel blocks do not line up with the 10 pixel cells.
    write_tile(folder / "t_Map.tif", data, -91.0, 15.0, block=16)
    pd.testing.assert_frame_equal(aggregate(folder, BOX), expected)


def test_tile_without_a_nodata_flag_still_treats_zero_as_no_data(tmp_path: Path) -> None:
    data = blank(value=40)
    paint(data, 0, 0, {40: 50, 0: 50})
    write_tile(tmp_path / "t_Map.tif", data, -91.0, 15.0, nodata=None)
    row = cell(aggregate(tmp_path, BOX), -90.975, 14.975)
    assert row["valid_fraction"] == pytest.approx(0.5)
    assert row["fraction_cropland"] == 1.0


def test_unknown_class_value_is_rejected(tmp_path: Path) -> None:
    data = blank()
    data[0, 0] = 77
    write_tile(tmp_path / "t_Map.tif", data, -91.0, 15.0)
    with pytest.raises(WorldCoverError, match="77"):
        aggregate(tmp_path, BOX)


def test_folder_without_tiles_is_reported(tmp_path: Path) -> None:
    with pytest.raises(WorldCoverError, match="No WorldCover tiles"):
        aggregate(tmp_path, BOX)


def test_box_outside_every_tile_is_rejected(tiles: Path) -> None:
    with pytest.raises(WorldCoverError, match="outside"):
        aggregate(tiles, (10.0, 10.0, 10.5, 10.5))


def test_tiles_with_different_pixel_sizes_are_rejected(tmp_path: Path) -> None:
    write_tile(tmp_path / "a_Map.tif", blank(), -91.0, 15.0)
    write_tile(tmp_path / "b_Map.tif", blank(50, 50), -90.5, 15.0, pixel=0.01)
    with pytest.raises(WorldCoverError, match="pixel size"):
        aggregate(tmp_path, (-91.0, 14.9, -90.4, 15.0))


def test_invalid_box_is_rejected(tiles: Path) -> None:
    with pytest.raises(WorldCoverError, match="box"):
        aggregate(tiles, (-90.5, 14.5, -91.0, 15.0))


def test_clean_output_has_table_geotiff_and_an_honest_manifest(tiles: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"
    manifest_path = write_clean(tiles, out, BOX, area_name="test area")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    table = pd.read_parquet(out / manifest["file"])

    assert len(table) == manifest["rows"] == 100
    assert manifest["dataset"] == "landcover_worldcover_cropland_mask"
    assert "ESA WorldCover" in manifest["source"]
    assert manifest["source_year"] == 2021
    assert manifest["source_version"] == "v200"
    assert "CC BY 4.0" in manifest["license"]
    assert manifest["attribution"] == (
        "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) "
        "processed by ESA WorldCover consortium"
    )
    assert manifest["grid"]["cell_degrees"] == 0.05
    assert manifest["grid"]["crs"] == "EPSG:4326"
    assert manifest["threshold"]["cropland_fraction"] == 0.5
    assert manifest["threshold"]["status"] == "team assumption, to be reviewed"
    assert manifest["classes"]["40"] == "cropland"
    assert manifest["bbox"] == {"west": -91.0, "south": 14.5, "east": -90.5, "north": 15.0}
    assert len(manifest["tiles"]) == 1
    tile = manifest["tiles"][0]
    assert tile["file"] == "ESA_WorldCover_10m_2021_v200_N14W091_Map.tif"
    assert len(tile["checksum_sha256"]) == 64
    assert any("which crop" in text for text in manifest["limitations"])
    assert any("2021" in text for text in manifest["limitations"])
    assert any("built-up" in text.lower() for text in manifest["limitations"])
    assert manifest["cropland_cells"] == int(table["cropland_mask"].sum()) == 2
    assert len(manifest["checksum_sha256"]) == 64
    assert len(manifest["raster"]["checksum_sha256"]) == 64

    with rasterio.open(out / manifest["raster"]["file"]) as grid:
        assert grid.profile["tiled"]
        assert grid.crs.to_string() == "EPSG:4326"
        assert grid.transform.a == pytest.approx(0.05)
        assert grid.bounds.left == pytest.approx(-91.0)
        assert grid.bounds.top == pytest.approx(15.0)
        data = grid.read(1)
    assert data.shape == (10, 10)
    assert data[0, 0] == pytest.approx(0.30)
    assert np.isnan(data[0, 2])


def test_manifest_lists_only_the_tiles_the_box_touches(tmp_path: Path) -> None:
    write_tile(tmp_path / "near_Map.tif", blank(), -91.0, 15.0)
    write_tile(tmp_path / "far_Map.tif", blank(), -80.0, 15.0)
    manifest = json.loads(write_clean(tmp_path, tmp_path / "o", BOX, "x").read_text("utf-8"))
    assert [t["file"] for t in manifest["tiles"]] == ["near_Map.tif"]


def test_command_line_writes_the_clean_dataset(
    tiles: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clean"
    bbox = ["-91.0", "14.5", "-90.5", "15.0"]
    code = main(
        ["worldcover", "--input", str(tiles), "--out", str(out), "--bbox", *bbox]
        + ["--area-name", "test area"]
    )
    assert code == 0
    assert (out / "landcover_worldcover_cropland_mask.parquet").is_file()
    printed = capsys.readouterr().out
    assert "100 cells" in printed
    assert "2 in the cropland mask" in printed


# ---- an empty mask is reported, never silent -------------------------------------------


@pytest.fixture
def grass_only(tmp_path: Path) -> Path:
    folder = tmp_path / "grass"
    folder.mkdir()
    data = blank()
    paint(data, 0, 0, {40: 20, GRASS: 80})  # some cropland, far below the threshold
    write_tile(folder / "ESA_WorldCover_10m_2021_v200_N14W091_Map.tif", data, -91.0, 15.0)
    return folder


def test_empty_mask_is_flagged_in_the_manifest_with_the_largest_fraction(
    grass_only: Path, tmp_path: Path
) -> None:
    out = tmp_path / "clean"
    manifest = json.loads(write_clean(grass_only, out, BOX, "grass").read_text("utf-8"))
    assert manifest["cropland_cells"] == 0
    assert manifest["max_cropland_fraction"] == pytest.approx(0.20)
    (warning,) = manifest["warnings"]
    assert "empty" in warning
    assert "does not mean" in warning
    assert "fraction" in warning


def test_nonempty_mask_has_no_warning(tiles: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"
    manifest = json.loads(write_clean(tiles, out, BOX, "ok").read_text("utf-8"))
    assert manifest["cropland_cells"] > 0
    assert manifest["warnings"] == []


def test_command_line_prints_the_empty_mask_warning(
    grass_only: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["worldcover", "--input", str(grass_only), "--out", str(tmp_path / "o"), "--bbox",
            "-91.0", "14.5", "-90.5", "15.0", "--area-name", "grass"]  # fmt: skip
    assert main(args) == 0
    assert "WARNING" in capsys.readouterr().err


# ---- cover inside the polygon of a farm --------------------------------------------------


def square(west: float, south: float, east: float, north: float) -> dict[str, object]:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


def test_polygon_cover_counts_the_pixels_inside_the_polygon(tiles: Path) -> None:
    # the first three pixel rows of cell (0, 0) are all cropland: 30 pixels
    cover = polygon_cover(tiles, square(-91.0, 14.985, -90.95, 15.0))
    assert cover.valid_pixels == 30
    assert cover.fractions["cropland"] == pytest.approx(1.0)
    assert cover.valid_fraction == pytest.approx(1.0)


def test_polygon_over_a_whole_cell_matches_the_cell_fractions(tiles: Path) -> None:
    cover = polygon_cover(tiles, square(-91.0, 14.95, -90.95, 15.0))
    assert cover.fractions["cropland"] == pytest.approx(0.30)
    assert cover.fractions["tree_cover"] == pytest.approx(0.20)
    assert cover.fractions["grassland"] == pytest.approx(0.50)


def test_polygon_over_no_data_gives_null_cropland(tiles: Path) -> None:
    cover = polygon_cover(tiles, square(-90.9, 14.95, -90.85, 15.0))
    assert cover.valid_pixels == 0
    assert cover.valid_fraction == 0
    assert cover.fractions["cropland"] is None


def test_polygon_crossing_two_tiles_counts_every_pixel_once(tmp_path: Path) -> None:
    folder = tmp_path / "two"
    folder.mkdir()
    write_tile(folder / "a_Map.tif", blank(), -91.0, 15.0)
    write_tile(folder / "b_Map.tif", blank(value=40), -90.5, 15.0)
    cover = polygon_cover(folder, square(-90.55, 14.9, -90.45, 14.95))
    assert cover.valid_pixels == 10 * 20
    assert cover.fractions["cropland"] == pytest.approx(0.5)


def test_polygon_accepts_a_feature_and_a_multipolygon(tiles: Path) -> None:
    whole = square(-91.0, 14.95, -90.95, 15.0)
    feature = {"type": "Feature", "properties": {}, "geometry": whole}
    multi = {"type": "MultiPolygon", "coordinates": [whole["coordinates"]]}
    assert polygon_cover(tiles, feature).valid_pixels == 100
    assert polygon_cover(tiles, multi).valid_pixels == 100


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"type": "Point", "coordinates": [-90.9, 14.9]}, "Polygon"),
        (square(10.0, 10.0, 10.1, 10.1), "outside"),
        (square(-91.0, 14.5, -90.0, 15.5), "too large"),
    ],
    ids=["point", "outside_tiles", "huge"],
)
def test_bad_polygons_are_rejected(tiles: Path, geometry: dict[str, object], message: str) -> None:
    with pytest.raises(WorldCoverError, match=message):
        polygon_cover(tiles, geometry)


def test_declared_cropland_that_the_product_does_not_see_gets_a_warning(tiles: Path) -> None:
    cover = polygon_cover(tiles, square(-90.65, 14.5, -90.6, 14.55))  # all grassland
    message = check_declared_cropland(cover, declared_cropland=True)
    assert message is not None
    assert "does not match" in message
    assert "does not replace the farmer" in message


def test_matching_or_undeclared_cropland_gets_no_warning(tiles: Path) -> None:
    crop = polygon_cover(tiles, square(-91.0, 14.985, -90.95, 15.0))
    grass = polygon_cover(tiles, square(-90.65, 14.5, -90.6, 14.55))
    assert check_declared_cropland(crop, declared_cropland=True) is None
    assert check_declared_cropland(grass, declared_cropland=False) is None
    assert LOW_CROPLAND_FRACTION < 0.5  # the polygon check is more lenient than the cell mask


def test_declared_cropland_over_no_data_says_there_is_nothing_to_compare(tiles: Path) -> None:
    cover = polygon_cover(tiles, square(-90.9, 14.95, -90.85, 15.0))
    message = check_declared_cropland(cover, declared_cropland=True)
    assert message is not None
    assert "no data" in message


def test_command_line_reports_the_polygon_cover(
    tiles: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    geojson = tmp_path / "farm.geojson"
    geojson.write_text(json.dumps(square(-90.65, 14.5, -90.6, 14.55)), encoding="utf-8")
    out = tmp_path / "cover.json"
    args = ["worldcover-polygon", "--input", str(tiles), "--polygon", str(geojson),
            "--out", str(out), "--declared-cropland"]  # fmt: skip
    assert main(args) == 0
    report = json.loads(out.read_text("utf-8"))
    assert report["fractions"]["grassland"] == pytest.approx(1.0)
    assert report["source_year"] == 2021
    assert "does not match" in report["warnings"][0]
    assert "WARNING" in capsys.readouterr().err
