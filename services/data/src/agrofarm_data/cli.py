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
    return 0
