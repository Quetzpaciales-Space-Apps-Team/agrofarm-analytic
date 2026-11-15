"""Command line entry point for the data jobs. Each connector adds its own subcommand."""

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from agrofarm_data import __version__
from agrofarm_data.connector import Connector, run_job
from agrofarm_data.connectors import (
    chirps,
    enso,
    insivumeh,
    meteomatics,
    nasa_power,
    planetary_computer,
    soilgrids,
    worldcover,
)
from agrofarm_data.terrain import pipeline as terrain


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

    power = commands.add_parser(
        "nasa-power", help="Clean a NASA POWER daily point CSV, or fetch point or regional data"
    )
    power.add_argument("--input", type=Path, help="Path to the POWER daily .csv (offline mode)")
    power.add_argument(
        "--fetch", choices=["point", "regional"],
        help="Download from the POWER API (community AG) instead of reading --input",
    )  # fmt: skip
    power.add_argument("--lat", type=float, help="Latitude for --fetch point")
    power.add_argument("--lon", type=float, help="Longitude for --fetch point")
    power.add_argument(
        "--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Area for --fetch regional: 2 to 10 degrees per axis, EPSG:4326",
    )  # fmt: skip
    power.add_argument("--start", type=dt.date.fromisoformat, help="YYYY-MM-DD, for --fetch")
    power.add_argument("--end", type=dt.date.fromisoformat, help="YYYY-MM-DD, for --fetch")
    power.add_argument("--raw", type=Path, help="Raw store directory (write once), for --fetch")
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

    relief = commands.add_parser(
        "terrain", help="Slope, aspect, flow accumulation and risk zones from Copernicus DEM tiles"
    )
    relief.add_argument("--input", type=Path, required=True, help="Folder with the DEM .tif tiles")
    relief.add_argument("--out", type=Path, required=True, help="Directory for the dataset")
    relief.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Area in EPSG:4326",
    )
    relief.add_argument(
        "--area-name", required=True, help="Name of the area, written to the manifest"
    )

    ndvi = commands.add_parser(
        "planetary-ndvi",
        help="NDVI series with a cloud mask from Microsoft Planetary Computer for a small box",
    )
    ndvi.add_argument(
        "--collection", required=True, choices=sorted(planetary_computer.SPECS),
        help="STAC collection (modis-13q1-061 is the STAC id modis-13Q1-061)",
    )  # fmt: skip
    ndvi.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Small area in EPSG:4326",
    )
    ndvi.add_argument("--start", required=True, help="First day, YYYY-MM-DD")
    ndvi.add_argument("--end", required=True, help="Last day, YYYY-MM-DD")
    ndvi.add_argument("--out", type=Path, required=True, help="Directory for the dataset")
    ndvi.add_argument(
        "--area-name", required=True, help="Name of the area, written to the manifest"
    )
    ndvi.add_argument(
        "--max-items", type=int, default=None, help="Keep this many scenes, spread over the dates"
    )
    ndvi.add_argument(
        "--max-cloud", type=float, default=100.0,
        help="Sentinel-2 only: skip scenes whose tile cloud cover is at or above this percent",
    )  # fmt: skip
    ndvi.add_argument(
        "--raw-dir", type=Path, default=None, help="Raw store folder (default: <out>/raw)"
    )

    weather = commands.add_parser(
        "meteomatics-forecast",
        help="Daily point forecast from Meteomatics (cached, with a daily request budget)",
    )
    weather.add_argument("--lat", type=float, required=True)
    weather.add_argument("--lon", type=float, required=True)
    weather.add_argument("--days", type=int, default=7)
    weather.add_argument("--cache", type=Path, required=True, help="Cache and budget directory")
    weather.add_argument("--out", type=Path, required=True, help="Directory for the clean table")
    weather.add_argument("--daily-limit", type=int, default=meteomatics.DEFAULT_DAILY_LIMIT)

    rain = commands.add_parser(
        "chirps", help="Download CHIRPS v3 daily rainfall for an area (about 18 MB per day)"
    )
    rain.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Area in EPSG:4326",
    )  # fmt: skip
    rain.add_argument("--start", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    rain.add_argument("--end", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    rain.add_argument("--raw", type=Path, required=True, help="Raw store directory (write once)")
    rain.add_argument("--out", type=Path, required=True, help="Directory for the clean dataset")
    return parser


def _check_ndvi_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        parser.error(
            "--bbox must be WEST SOUTH EAST NORTH in degrees with west < east, south < north"
        )
    try:
        first, last = date.fromisoformat(args.start), date.fromisoformat(args.end)
    except ValueError:
        parser.error("--start and --end must be YYYY-MM-DD")
    if first > last:
        parser.error("--start must not be after --end")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be at least 1")


def _check_power_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.input is None) == (args.fetch is None):
        parser.error("give exactly one of --input (a CSV) or --fetch (point or regional)")
    if args.fetch is None:
        return
    if args.start is None or args.end is None or args.raw is None:
        parser.error("--fetch needs --start, --end and --raw")
    if args.fetch == "point" and (args.lat is None or args.lon is None):
        parser.error("--fetch point needs --lat and --lon")
    if args.fetch == "regional" and args.bbox is None:
        parser.error("--fetch regional needs --bbox WEST SOUTH EAST NORTH")


def main(
    argv: Sequence[str] | None = None,
    *,
    http: planetary_computer.HttpJson = planetary_computer.http_json,
    reader: planetary_computer.RasterReader = planetary_computer.read_window,
    power_transport: nasa_power.Transport | None = None,
) -> int:
    """Run a command. `http`, `reader` and `power_transport` let tests replace the network."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "nasa-power":
        _check_power_args(parser, args)
    if args.command == "planetary-ndvi":
        _check_ndvi_args(parser, args)
    if args.command == "insivumeh":
        frame = insivumeh.read_insivumeh(args.input)
        manifest = insivumeh.write_clean(frame, args.out, source_file=args.input)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "enso":
        frame = enso.read_enso(args.oni, args.roni)
        manifest = enso.write_clean(frame, args.out, oni_file=args.oni, roni_file=args.roni)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "nasa-power" and args.fetch is not None:
        send = power_transport or nasa_power._download
        with tempfile.TemporaryDirectory() as downloads:
            try:
                if args.fetch == "point":
                    power_job: Connector = nasa_power.PowerPointConnector(
                        args.lat, args.lon, args.start, args.end, Path(downloads), send
                    )
                else:
                    west, south, east, north = args.bbox
                    power_job = nasa_power.PowerRegionalConnector(
                        south, north, west, east, args.start, args.end, Path(downloads), send
                    )
                manifest = run_job(power_job, args.raw, args.out)
            except nasa_power.PowerError as error:
                sys.stderr.write(f"ERROR: {error}\n")
                return 2
        sys.stdout.write(f"{json.loads(manifest.read_text('utf-8'))['rows']} rows -> {manifest}\n")
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
    elif args.command == "terrain":
        manifest = terrain.write_clean(args.input, args.out, tuple(args.bbox), args.area_name)
        count = len(json.loads(manifest.read_text("utf-8"))["layers"])
        sys.stdout.write(f"{count} layers -> {manifest}\n")
    elif args.command == "planetary-ndvi":
        manifest = planetary_computer.run_planetary_ndvi(
            args.collection, tuple(args.bbox), args.start, args.end, args.out, args.area_name,
            max_items=args.max_items, max_cloud=args.max_cloud, raw_dir=args.raw_dir,
            http=http, reader=reader,
        )  # fmt: skip
        info = json.loads(manifest.read_text("utf-8"))
        with_ndvi = sum(1 for x in info["scenes"] if x["status"] == "ok")
        sys.stdout.write(
            f"{info['rows']} scenes (found {info['scenes_found']}), {with_ndvi} with NDVI "
            f"-> {manifest}\n"
        )
        for text in info["warnings"]:
            sys.stderr.write(f"WARNING: {text}\n")
    elif args.command == "meteomatics-forecast":
        api = meteomatics.MeteomaticsClient(
            cache_dir=args.cache,
            username=os.environ.get("METEOMATICS_USERNAME", ""),
            password=os.environ.get("METEOMATICS_PASSWORD", ""),
            daily_limit=args.daily_limit,
        )
        try:
            frame = api.forecast(args.lat, args.lon, days=args.days)
        except meteomatics.MeteomaticsError as error:
            sys.stderr.write(f"ERROR: {error}\n")
            return 2
        manifest = meteomatics.write_clean(frame, args.out, api.last_response_file)
        used = api.requests_today()
        sys.stdout.write(f"{len(frame)} days -> {manifest} ({used} requests used today)\n")
    elif args.command == "chirps":
        with tempfile.TemporaryDirectory() as downloads:
            try:
                job = chirps.ChirpsConnector(
                    tuple(args.bbox), args.start, args.end, Path(downloads),
                    transport=lambda url: chirps._download(url),
                )  # fmt: skip
                manifest = run_job(job, args.raw, args.out)
            except chirps.ChirpsError as error:
                sys.stderr.write(f"ERROR: {error}\n")
                return 2
        rows = json.loads(manifest.read_text("utf-8"))["rows"]
        sys.stdout.write(f"{rows} rows -> {manifest}\n")
    return 0
