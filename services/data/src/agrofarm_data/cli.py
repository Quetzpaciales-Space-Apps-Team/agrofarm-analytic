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
from agrofarm_data.chirps_job import DirectoryUploader, run_chirps_job
from agrofarm_data.connector import Connector, run_job
from agrofarm_data.connectors import (
    appeears,
    chirps,
    enso,
    insivumeh,
    meteomatics,
    nasa_power,
    planetary_computer,
    rio_contreras,
    soilgrids,
    worldcover,
)
from agrofarm_data.gcs_storage import GcsStorage
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

    contreras = commands.add_parser(
        "rio-contreras", help="Ingest one reviewed Rio Contreras rainfall source"
    )
    contreras.add_argument(
        "--source",
        choices=sorted(rio_contreras.SPECS),
        required=True,
        help="Reviewed source type; each source writes a separate clean dataset",
    )
    contreras.add_argument("--input", type=Path, required=True, help="Source .xlsx or .csv file")
    contreras.add_argument("--out", type=Path, required=True, help="Directory for the clean table")
    contreras.add_argument(
        "--raw-dir", type=Path, required=True, help="Write-once raw storage directory"
    )

    indices = commands.add_parser("enso", help="Clean the NOAA ONI and RONI text files")
    indices.add_argument("--oni", type=Path, required=True, help="Path to oni.ascii.txt")
    indices.add_argument("--roni", type=Path, required=True, help="Path to RONI.ascii.txt")
    indices.add_argument("--out", type=Path, required=True, help="Directory for the clean table")

    power = commands.add_parser(
        "nasa-power", help="Clean a NASA POWER daily point CSV, or fetch point or regional data"
    )
    power.add_argument("--input", type=Path, help="Path to the POWER daily .csv (offline mode)")
    power.add_argument("--fetch", choices=["point", "regional"])
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

    earth = commands.add_parser(
        "appeears",
        help="SMAP L4 soil moisture, MODIS or VIIRS vegetation for a box, through NASA AppEEARS",
        description="Submits an AppEEARS task on the first run and exits with code 75 while it "
        "runs; run the same command again to collect it. Needs EARTHDATA_USERNAME and "
        "EARTHDATA_PASSWORD in the environment (AppEEARS does not accept the profile token).",
    )
    earth.add_argument("--source", choices=sorted(appeears.SPECS), required=True)
    earth.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Area in EPSG:4326",
    )  # fmt: skip
    earth.add_argument("--start", required=True, help="YYYY-MM-DD")
    earth.add_argument("--end", required=True, help="YYYY-MM-DD")
    earth.add_argument("--raw-dir", type=Path, required=True, help="Write-once raw storage")
    earth.add_argument("--out", type=Path, required=True, help="Directory for the clean table")
    earth.add_argument(
        "--state-dir", type=Path, default=None, help="Task ids (default: <raw-dir>/appeears-state)"
    )
    earth.add_argument(
        "--wait-seconds", type=float, default=0.0,
        help="Keep polling for this long before exiting with 75 (default: do not wait)",
    )  # fmt: skip
    earth.add_argument("--poll-seconds", type=float, default=30.0)
    earth.add_argument(
        "--min-usable-fraction", type=float, default=appeears.MIN_USABLE_FRACTION,
        help="Vegetation only: share of usable pixels a composite needs to keep its values "
        "(default 0.5; in the rainy season a lower value keeps more, noisier dates)",
    )  # fmt: skip

    rain = commands.add_parser(
        "chirps", help="Read CHIRPS rainfall windows and write a monthly dataset"
    )
    rain.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Area in EPSG:4326",
    )  # fmt: skip
    rain.add_argument("--start", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    rain.add_argument("--end", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    rain.add_argument("--out", type=Path, help="Local clean output directory")
    rain.add_argument(
        "--raw-dir", "--raw", dest="raw_dir", type=Path, help="Local immutable raw store"
    )
    rain.add_argument(
        "--gcs", action="store_true", help="Run as a Cloud Run Job and upload to Cloud Storage"
    )
    rain.add_argument("--raw-bucket", help="Raw bucket (default: GCS_BUCKET_RAW)")
    rain.add_argument("--clean-bucket", help="Clean bucket (default: GCS_BUCKET_CLEAN)")
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


def _check_appeears_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        appeears.build_request(
            appeears.SPECS[args.source], tuple(args.bbox), args.start, args.end, "check"
        )
    except appeears.AppEearsError as error:
        parser.error(str(error))
    if not 0 < args.min_usable_fraction <= 1:
        parser.error("--min-usable-fraction must be above 0 and at most 1")


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
    chirps_window_reader: chirps.WindowReader | None = None,
    gcs_storage: DirectoryUploader | None = None,
    appeears_transport: appeears.Transport | None = None,
) -> int:
    """Run a command with injectable HTTP and raster readers for offline tests."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "nasa-power":
        _check_power_args(parser, args)
    if args.command == "chirps":
        _check_chirps_args(parser, args)
    if args.command == "planetary-ndvi":
        _check_ndvi_args(parser, args)
    if args.command == "appeears":
        _check_appeears_args(parser, args)
    if args.command == "insivumeh":
        frame = insivumeh.read_insivumeh(args.input)
        manifest = insivumeh.write_clean(frame, args.out, source_file=args.input)
        sys.stdout.write(f"{len(frame)} rows -> {manifest}\n")
    elif args.command == "rio-contreras":
        rio_connector = rio_contreras.make_connector(args.source, args.input)
        manifest = run_job(rio_connector, args.raw_dir, args.out)
        rows = json.loads(manifest.read_text("utf-8"))["rows"]
        sys.stdout.write(f"{rows} rows -> {manifest}\n")
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
        rows = json.loads(manifest.read_text("utf-8"))["rows"]
        sys.stdout.write(f"{rows} rows -> {manifest}\n")
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
    elif args.command == "appeears":
        return _run_appeears(args, appeears_transport)
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
        start, end = args.start, args.end
        bbox = tuple(args.bbox)
        if args.gcs:
            raw_bucket = args.raw_bucket or os.environ.get("GCS_BUCKET_RAW")
            clean_bucket = args.clean_bucket or os.environ.get("GCS_BUCKET_CLEAN")
            if not raw_bucket or not clean_bucket:
                parser.error("--gcs requires both bucket flags or GCS_BUCKET_RAW/GCS_BUCKET_CLEAN")
            manifest_uri = run_chirps_job(
                bbox=bbox,
                start=start,
                end=end,
                raw_bucket=raw_bucket,
                clean_bucket=clean_bucket,
                window_reader=chirps_window_reader or chirps._read_remote_window,
                uploader=gcs_storage or GcsStorage(),
            )  # fmt: skip
            sys.stdout.write(f"CHIRPS monthly dataset uploaded -> {manifest_uri}\n")
        else:
            if args.out is None:
                parser.error("local chirps mode requires --out")
            raw_dir = args.raw_dir or args.out / "raw"
            connector = chirps.ChirpsConnector(
                bbox,
                start,
                end,
                raw_dir / chirps.DATASET / chirps.VERSION,
                window_reader=chirps_window_reader or chirps._read_remote_window,
            )
            manifest = run_job(connector, raw_dir, args.out)
            info = json.loads(manifest.read_text("utf-8"))
            sys.stdout.write(f"{info['rows']} cells -> {manifest}\n")
    return 0


def _run_appeears(args: argparse.Namespace, transport: appeears.Transport | None) -> int:
    """Exit codes: 0 collected, 75 the task is still running (run again), 2 an error."""
    connector: appeears.AppEearsConnector | None = None
    try:
        username, password = appeears.credentials_from_env(os.environ)
        api = transport or appeears.UrllibTransport()
        token = appeears.login(api, username, password)
        spec = appeears.SPECS[args.source]
        bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
        connector = appeears.AppEearsConnector(
            spec, bbox, args.start, args.end,
            transport=api, token=token,
            state_dir=args.state_dir or args.raw_dir / "appeears-state",
            area_name=f"agrofarm-{spec.key}-{args.start}-{args.end}",
            wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds,
            min_usable_fraction=args.min_usable_fraction,
        )  # fmt: skip
        manifest = run_job(connector, args.raw_dir, args.out)
    except appeears.TaskPending as pending:
        sys.stderr.write(f"{pending}\n")
        return 75
    except appeears.AppEearsError as error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 2
    finally:
        if connector is not None:
            connector.cleanup()
    info = json.loads(manifest.read_text("utf-8"))
    sys.stdout.write(f"{info['rows']} rows ({info['product']}) -> {manifest}\n")
    return 0


def _check_chirps_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        chirps._grid(tuple(args.bbox))
        first, last = args.start, args.end
        if first > last or first < date(1981, 1, 1):
            raise ValueError
        if args.gcs and (first.year, first.month) != (last.year, last.month):
            parser.error("Cloud Run CHIRPS executions must cover one calendar month")
        if not args.gcs and args.out is None:
            parser.error("local chirps mode requires --out")
    except ValueError:
        parser.error("CHIRPS requires a valid bbox and ordered YYYY-MM-DD dates")
