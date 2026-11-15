"""Command line entry point for the data jobs. Each connector adds its own subcommand."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agrofarm_data import __version__
from agrofarm_data.connectors import enso, insivumeh, nasa_power, soilgrids, worldcover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agrofarm-data", description="AgroFarm data-service jobs: fetch, clean, register."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    station = commands.add_parser("insivumeh", help="Clean the INSIVUMEH daily climate workbook")
    station.add_argument("--input", type=Path, required=True, help="Path to the .xlsx workbook")
    station.add_argument("--out", type=Path, required=True, help="Directory for the clean table")

    indices = commands.add_parser("enso", help="Clean the NOAA ONI and RONI text files")
    indices.add_argument("--oni", type=Path, required=True, help="Path to oni.ascii.txt")
    indices.add_argument("--roni", type=Path, required=True, help="Path to RONI.ascii.txt")
    indices.add_argument("--out", type=Path, required=True, help="Directory for the clean table")

    power = commands.add_parser("nasa-power", help="Clean a NASA POWER daily point CSV")
    power.add_argument("--input", type=Path, required=True, help="Path to the POWER daily .csv")
    power.add_argument("--out", type=Path, required=True, help="Directory for the clean table")

    soil = commands.add_parser("soilgrids", help="Crop and scale the SoilGrids layers for an area")
    soil.add_argument(
        "--input", type=Path, required=True, help="Folder with <property>_<depth>_mean.tif"
    )
    soil.add_argument("--out", type=Path, required=True, help="Directory for the clean dataset")
    soil.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Area in EPSG:4326",
    )
    soil.add_argument(
        "--area-name", required=True, help="Name of the area, written to the manifest"
    )

    cover = commands.add_parser(
        "worldcover", help="Cropland mask on the 0.05 degree grid from ESA WorldCover tiles"
    )
    cover.add_argument(
        "--input", type=Path, required=True, help="Folder with ESA_WorldCover_*_Map.tif tiles"
    )
    cover.add_argument("--out", type=Path, required=True, help="Directory for the clean dataset")
    cover.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Area in EPSG:4326",
    )
    cover.add_argument(
        "--area-name", required=True, help="Name of the area, written to the manifest"
    )

    farm = commands.add_parser(
        "worldcover-polygon", help="Land cover shares inside a farm polygon from 10 m pixels"
    )
    farm.add_argument(
        "--input", type=Path, required=True, help="Folder with ESA_WorldCover_*_Map.tif tiles"
    )
    farm.add_argument("--polygon", type=Path, required=True, help="GeoJSON Polygon, EPSG:4326")
    farm.add_argument("--out", type=Path, required=True, help="JSON file for the report")
    farm.add_argument(
        "--declared-cropland", action="store_true", help="The farm declares that it grows crops"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "insivumeh":
        frame = insivumeh.read_insivumeh(args.input)
        manifest = insivumeh.write_clean(frame, args.out, source_file=args.input)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "enso":
        frame = enso.read_enso(args.oni, args.roni)
        manifest = enso.write_clean(frame, args.out, oni_file=args.oni, roni_file=args.roni)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "nasa-power":
        frame = nasa_power.read_power(args.input)
        manifest = nasa_power.write_clean(frame, args.out, source_file=args.input)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "soilgrids":
        manifest = soilgrids.write_clean(args.input, args.out, tuple(args.bbox), args.area_name)
        rows = json.loads(manifest.read_text("utf-8"))["rows"]
        sys.stdout.write(f"{rows} rows -> {manifest}\n")
    elif args.command == "worldcover":
        manifest = worldcover.write_clean(args.input, args.out, tuple(args.bbox), args.area_name)
        info = json.loads(manifest.read_text("utf-8"))
        sys.stdout.write(
            f"{info['rows']} cells, {info['cropland_cells']} in the cropland mask -> {manifest}\n"
        )
        for text in info["warnings"]:
            sys.stderr.write(f"WARNING: {text}\n")
    elif args.command == "worldcover-polygon":
        geometry = json.loads(args.polygon.read_text("utf-8"))
        result = worldcover.polygon_cover(args.input, geometry)
        message = worldcover.check_declared_cropland(result, args.declared_cropland)
        report = {
            "source": "ESA WorldCover 2021 v200 (10 m)",
            "source_year": worldcover.SOURCE_YEAR,
            "attribution": worldcover.ATTRIBUTION,
            "fractions": result.fractions,
            "valid_pixels": result.valid_pixels,
            "valid_fraction": round(result.valid_fraction, 6),
            "warnings": [message] if message else [],
            "limitations": worldcover.LIMITATIONS,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        cropland = result.fractions["cropland"]
        shown = "no data" if cropland is None else f"{cropland:.1%}"
        sys.stdout.write(f"cropland {shown} of {result.valid_pixels} valid pixels -> {args.out}\n")
        if message:
            sys.stderr.write(f"WARNING: {message}\n")
    return 0
