"""Every existing connector writes a manifest that validates against the schema.

The inputs are the synthetic fixtures built by the connector tests. The connectors are not
refactored onto the `Connector` interface here; only their manifests are checked.
"""

from pathlib import Path
from typing import Any

import pytest

from agrofarm_data.connectors import enso, insivumeh, nasa_power, soilgrids, worldcover
from agrofarm_data.manifest import validate_manifest
from agrofarm_data.terrain import pipeline as terrain
from tests import test_soilgrids as soil_tests
from tests import test_terrain as terrain_tests
from tests import test_worldcover as cover_tests
from tests.test_insivumeh import workbook  # noqa: F401
from tests.test_nasa_power import source as power_source  # noqa: F401
from tests.test_terrain import tiles as dem_tiles  # noqa: F401
from tests.test_worldcover import tiles as cover_tiles  # noqa: F401

FIXTURES = Path(__file__).parent / "fixtures"


def check(manifest_path: Path) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    assert manifest["source_version"]
    assert manifest["attribution"]
    return manifest


def test_enso_manifest(tmp_path: Path) -> None:
    oni, roni = FIXTURES / "oni_sample.txt", FIXTURES / "roni_sample.txt"
    check(enso.write_clean(enso.read_enso(oni, roni), tmp_path, oni_file=oni, roni_file=roni))


def test_insivumeh_manifest(workbook: Path, tmp_path: Path) -> None:  # noqa: F811
    frame = insivumeh.read_insivumeh(workbook)
    manifest = check(insivumeh.write_clean(frame, tmp_path, source_file=workbook))
    assert manifest["preliminary"] is True


def test_nasa_power_manifest(power_source: Path, tmp_path: Path) -> None:  # noqa: F811
    frame = nasa_power.read_power(power_source)
    check(nasa_power.write_clean(frame, tmp_path, source_file=power_source))


def test_soilgrids_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "soilgrids"
    raw.mkdir()
    for prop, value in soil_tests.RAW.items():
        for depth in soilgrids.DEPTHS:
            soil_tests.write_raster(raw / f"{prop}_{depth}_mean.tif", value, zero_cells=4)
    manifest = check(soilgrids.write_clean(raw, tmp_path / "out", soil_tests.BBOX, "test area"))
    assert len(manifest["source_checksum_sha256"]) == len(soil_tests.RAW) * len(soilgrids.DEPTHS)


def test_worldcover_manifest(cover_tiles: Path, tmp_path: Path) -> None:  # noqa: F811
    path = worldcover.write_clean(cover_tiles, tmp_path, cover_tests.BOX, "test area")
    manifest = check(path)
    assert list(manifest["source_checksum_sha256"]) == [t["file"] for t in manifest["tiles"]]


def test_terrain_manifest(dem_tiles: Path, tmp_path: Path) -> None:  # noqa: F811
    path = terrain.write_clean(dem_tiles, tmp_path, terrain_tests.BOX, "test area")
    manifest = check(path)
    assert set(manifest["source_checksum_sha256"]) == {t["file"] for t in manifest["tiles"]}
    assert manifest["license_verified"] is False


@pytest.mark.parametrize("module", [enso, insivumeh, nasa_power])
def test_table_connectors_name_a_dataset_version(module: Any) -> None:
    assert module.SOURCE_VERSION
    assert module.ATTRIBUTION
