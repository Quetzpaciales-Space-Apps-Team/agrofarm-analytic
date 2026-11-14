"""Command line entry point for the data jobs. Each connector adds its own subcommand."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from agrofarm_data import __version__
from agrofarm_data.connectors import enso, insivumeh


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
    return 0
