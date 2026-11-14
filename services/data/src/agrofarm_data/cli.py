"""Command line entry point for the data jobs. Each connector adds its own subcommand."""

import argparse
from collections.abc import Sequence

from agrofarm_data import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agrofarm-data", description="AgroFarm data-service jobs: fetch, clean, register."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", required=True, metavar="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
