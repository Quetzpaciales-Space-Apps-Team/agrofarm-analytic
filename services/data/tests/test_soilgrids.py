"""SoilGrids connector: int16 rasters in conventional units, 0 means no data, scaled to
the units agronomists use, cropped to a bounding box, one table row per property and depth."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from agrofarm_data.cli import main
from agrofarm_data.connectors.soilgrids import (
    DEPTHS,
    PROPERTIES,
    SoilGridsError,
    read_layer,
    summarize,
    write_clean,
)

# A 4 x 4 raster covering lon -90.6..-90.4 and lat 14.4..14.6, cells of 0.05 degrees.
WEST, NORTH, CELL = -90.6, 14.6, 0.05
BBOX = (-90.6, 14.4, -90.4, 14.6)
RAW = {"phh2o": 62, "soc": 400, "clay": 300, "sand": 350, "silt": 350, "nitrogen": 300,
       "cec": 250, "bdod": 120}  # fmt: skip


def write_raster(path: Path, value: int, zero_cells: int = 0) -> None:
    data = np.full((4, 4), value, dtype="int16")
    data.flat[:zero_cells] = 0
    profile = {"driver": "GTiff", "height": 4, "width": 4, "count": 1, "dtype": "int16",
               "crs": "EPSG:4326", "transform": from_origin(WEST, NORTH, CELL, CELL)}  # fmt: skip
    with rasterio.open(path, "w", **profile) as out:
        out.write(data, 1)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "soilgrids"
    folder.mkdir()
    for prop, value in RAW.items():
        for depth in DEPTHS:
            write_raster(folder / f"{prop}_{depth}_mean.tif", value, zero_cells=4)
    return folder


def test_every_property_is_scaled_to_agronomic_units(raw_dir: Path) -> None:
    table = summarize(raw_dir, BBOX)
    top = table[table["depth"] == "0-5cm"].set_index("property")["mean"]
    assert top["phh2o"] == pytest.approx(6.2)  # pH x 10
    assert top["soc"] == pytest.approx(40.0)  # dg/kg to g/kg
    assert top["clay"] == pytest.approx(30.0)  # g/kg to %
    assert top["sand"] == pytest.approx(35.0)
    assert top["silt"] == pytest.approx(35.0)
    assert top["nitrogen"] == pytest.approx(3.0)  # cg/kg to g/kg
    assert top["cec"] == pytest.approx(25.0)  # mmol(c)/kg to cmol(c)/kg
    assert top["bdod"] == pytest.approx(1.2)  # cg/cm3 to kg/dm3


def test_one_row_per_property_and_depth(raw_dir: Path) -> None:
    table = summarize(raw_dir, BBOX)
    assert len(table) == len(PROPERTIES) * len(DEPTHS)
    assert not table.duplicated(["property", "depth"]).any()


def test_zero_is_no_data_not_a_value(raw_dir: Path) -> None:
    table = summarize(raw_dir, BBOX).set_index(["property", "depth"])
    key = ("phh2o", "0-5cm")
    assert table.at[key, "valid_fraction"] == pytest.approx(12 / 16)
    assert table.at[key, "mean"] == pytest.approx(6.2)  # not dragged down by the zeros
    assert table.at[key, "cells"] == 16


def test_area_without_data_gives_nulls_instead_of_zero(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    for prop in PROPERTIES:
        for depth in DEPTHS:
            write_raster(folder / f"{prop}_{depth}_mean.tif", 0)
    table = summarize(folder, BBOX)
    assert table["mean"].isna().all()
    assert (table["valid_fraction"] == 0).all()


def test_crop_keeps_only_cells_inside_the_box(raw_dir: Path) -> None:
    values, transform = read_layer(
        raw_dir / "phh2o_0-5cm_mean.tif", "phh2o", (-90.6, 14.5, -90.5, 14.6)
    )
    assert values.shape == (2, 2)
    assert transform.c == pytest.approx(-90.6)


def test_box_outside_the_raster_is_rejected(raw_dir: Path) -> None:
    with pytest.raises(SoilGridsError, match="outside"):
        summarize(raw_dir, (10.0, 10.0, 10.5, 10.5))


def test_missing_layer_is_reported_by_name(raw_dir: Path) -> None:
    (raw_dir / "soc_30-60cm_mean.tif").unlink()
    with pytest.raises(SoilGridsError, match="soc_30-60cm_mean.tif"):
        summarize(raw_dir, BBOX)


def test_unexpected_value_range_is_rejected(raw_dir: Path) -> None:
    write_raster(raw_dir / "phh2o_0-5cm_mean.tif", 300)  # pH 30
    with pytest.raises(SoilGridsError, match="phh2o"):
        summarize(raw_dir, BBOX)


def test_clean_output_has_table_cogs_and_an_honest_manifest(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "clean"
    manifest_path = write_clean(raw_dir, out, BBOX, area_name="test area")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    table = pd.read_parquet(out / manifest["file"])

    assert len(table) == manifest["rows"] == 48
    assert "250 m" in manifest["spatial_resolution"]
    assert "prediction" in manifest["limitations"][0].lower()
    assert any("soil analysis" in text for text in manifest["limitations"])
    assert any("0" in text and "no data" in text for text in manifest["limitations"])
    assert manifest["bbox"] == {"west": -90.6, "south": 14.4, "east": -90.4, "north": 14.6}
    assert manifest["units"]["phh2o"] == "pH"
    assert len(manifest["layers"]) == 48
    layer = out / "layers" / "phh2o_0-5cm.tif"
    assert layer.is_file()
    with rasterio.open(layer) as cog:
        assert cog.driver == "GTiff"
        assert cog.profile["tiled"]
        data = cog.read(1)
        assert np.isnan(data).sum() == 4
        assert np.nanmean(data) == pytest.approx(6.2)


def test_command_line_writes_the_clean_dataset(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clean"
    bbox = ["-90.6", "14.4", "-90.4", "14.6"]
    args = ["soilgrids", "--input", str(raw_dir), "--out", str(out), "--bbox", *bbox]
    code = main([*args, "--area-name", "test area"])
    assert code == 0
    assert (out / "soil_soilgrids_static.parquet").is_file()
    assert "48 rows" in capsys.readouterr().out
