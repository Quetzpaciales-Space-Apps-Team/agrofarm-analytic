# data-service

Connectors that turn public sources into clean, documented datasets. Run `make check` before a pull request.

## Layout

```
raw/<dataset>/<version>/<name>      source files, written once
raw/<dataset>/<version>/<name>.sha256
clean/<dataset>.parquet             tables (Parquet)
clean/layers/*.tif                  rasters (tiled GeoTIFF, EPSG:4326)
clean/<dataset>.manifest.json       one manifest per dataset
```

- `raw/` is write-once (`agrofarm_data.raw_store.store_raw`). Files are read-only (0o444) with a sha256 sidecar.
  Storing the same content again does nothing. The same name with different content raises
  `RawModifiedError`: put a new source release in a new `<version>` folder.
- `clean/` is derived and can be rebuilt from `raw/`.
- Every output has a manifest, checked against `src/agrofarm_data/manifest.schema.json` by
  `validate_manifest`. It requires dataset, file or artifacts, source, source version or retrieval date,
  license (and attribution for CC BY licenses), date range or `time_step` starting with `static`,
  spatial resolution, `checksum_sha256` of the output, `source_checksum_sha256` of the raw input
  (or `no_raw_input: true`) and processing version.

## Planetary Computer NDVI (DS-031)

```
uv run python -m agrofarm_data planetary-ndvi --collection sentinel-2-l2a|modis-13q1-061 \
  --bbox W S E N --start YYYY-MM-DD --end YYYY-MM-DD --out <dir> --area-name <name> [--max-items N]
```

Searches the public Planetary Computer STAC API (standard library only, no key), reads a window of the
cloud-optimized GeoTIFFs and writes a Parquet NDVI series (one row per scene, null NDVI when the valid
share is below 25 percent), one tiled NDVI GeoTIFF per scene in `layers/` and a manifest. Raw windows and
the STAC response go to `<out>/raw/` (write-once). Collection ids and assets: `sentinel-2-l2a` (B04, B08, SCL)
and `modis-13Q1-061` (`250m_16_days_NDVI`, `250m_16_days_pixel_reliability`). The cloud mask rules and the
scale factors are in each manifest. Tests use recorded fixtures in `tests/fixtures/planetary_computer/`.

## AppEEARS: SMAP L4, MODIS and VIIRS (DS-030)

```
EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=... \
uv run python -m agrofarm_data appeears --source smap-l4|modis-13q1|viirs-vnp13a1 \
  --bbox W S E N --start YYYY-MM-DD --end YYYY-MM-DD --raw-dir <dir> --out <dir> [--wait-seconds N]
```

NASA AppEEARS samples a product for a polygon and returns a bundle. A task takes minutes to an hour, so the
command never blocks by default: the first run submits the task and stores its id in `<raw-dir>/appeears-state/`,
and it **exits with code 75** while the task runs. Run the same command again to poll it once; when the task
is `done` it downloads the statistics files and writes the dataset. `--wait-seconds` keeps polling that long.

| `--source` | Product (confirmed against the catalog) | Output, one row per |
|---|---|---|
| `smap-l4` | `SPL4SMGP.008`: surface and root zone soil moisture, m3/m3 | day: count-weighted mean of the 3-hourly samples |
| `modis-13q1` | `MOD13Q1.061`: NDVI and EVI, 250 m | 16-day composite |
| `viirs-vnp13a1` | `VNP13A1.002`: NDVI and EVI, 500 m | 16-day composite |

Rules that matter when reading the tables:

- **The statistics include cloudy pixels.** A vegetation composite has values only when at least half of its
  pixels are in a usable reliability class (MODIS 0 and 1, VIIRS 0, 1 and 2); otherwise the values are null and
  `status` is `too_cloudy`. A SMAP day has a value only with 6 of its 8 samples. The manifest records the rule.
  In the rainy season this can leave a whole month empty: all six VIIRS composites of June 2025 for the
  reference box had between 1 and 49 percent usable pixels. `--min-usable-fraction` lowers the share (the
  manifest records the value used); the dates it keeps are noisier.
- Only polygon statistics are kept (a `*-Statistics.csv` of the bundle), not the per-date rasters, and the
  polygon is a box, not a farm.
- Each downloaded file is checked against the SHA-256 in the bundle listing. The bundle's `*-request.json`
  and the signed file URLs are never kept, because AppEEARS puts the account name in them.
- **Credentials:** AppEEARS does **not** accept the token of the Earthdata profile page (it answers 403). It
  issues its own session token for the Earthdata user name and password, read from `EARTHDATA_USERNAME` and
  `EARTHDATA_PASSWORD`. The session token lives in memory for one run; nothing secret is logged or stored.
  The account must have accepted the AppEEARS terms once at https://appeears.earthdatacloud.nasa.gov.

Tests use the answers recorded from the real API (`tests/fixtures/appeears/`, see its README).

## CHIRPS rainfall (DS-021)

Run a local monthly ingestion with:

```sh
uv run python -m agrofarm_data chirps --bbox WEST SOUTH EAST NORTH \
  --start YYYY-MM-DD --end YYYY-MM-DD --out <clean-dir> [--raw-dir <raw-dir>]
```

For a Cloud Run Job, pass `--gcs` and set `GCS_BUCKET_RAW` and `GCS_BUCKET_CLEAN` on the job, or
provide `--raw-bucket` and `--clean-bucket`. The start and end dates must be within one calendar
month. Example execution after creating a job from the data-service image:

```sh
gcloud run jobs execute data-service-staging --region <region> \
  --args=chirps,--bbox,-90.60,14.40,-90.40,14.60,--start,2024-01-01,--end,2024-01-31,--gcs \
  --update-env-vars=GCS_BUCKET_RAW=<raw-bucket>,GCS_BUCKET_CLEAN=<clean-bucket> --wait
```

The job reads the requested cells from CHIRPS v3 daily final RNL GeoTIFFs using HTTP byte ranges.
It stores the clipped source windows immutably under
`raw/climate_chirps_daily/v3.0-final-rnl/YYYY/MM/` and publishes the Parquet table, daily COGs and
manifest under `clean/climate_chirps_daily/YYYY/MM/`. It does not persist the global source rasters.
The CHIRPS GeoTIFFs are strip-organized, so HTTP range support avoids transferring unrelated
longitude columns but transfer volume still grows with the number of source rows crossed by the
requested area. The job uses the Cloud Run service account identity to upload; grant it object
create and get permissions on the raw bucket (raw conflict verification), and object create
permissions on the clean bucket; clean-bucket overwrite also needs object delete permission.
No service-account key is required. Local tests use the recorded window fixture and do not call
CHIRPS or Google Cloud.

## Rio Contreras rainfall (DS-015)

The reviewed FUNCAGUA / Meteo Tech files are ingested offline, one source per run. Each source writes
to a separate dataset; blank rainfall rows are omitted, and Zona 16 records for May and June 2025 are
excluded. The raw input is preserved in the write-once raw store. Use the source name that matches
the input file:

```sh
uv run python -m agrofarm_data rio-contreras --source funcagua-monthly \
  --input <Serie_mensual_Rio_Contreras.xlsx> --raw-dir <raw-dir> --out <clean-dir>
uv run python -m agrofarm_data rio-contreras --source cayala-daily \
  --input <CIT_UVG_Cayala_serie_completa.xlsx> --raw-dir <raw-dir> --out <clean-dir>
uv run python -m agrofarm_data rio-contreras --source zona16-daily \
  --input <Zona16_CHIRPS_analisis.xlsx> --raw-dir <raw-dir> --out <clean-dir>
uv run python -m agrofarm_data rio-contreras --source chirps-monthly \
  --input <chirps3_precip_monthly_14.604_-90.490_serie.csv> --raw-dir <raw-dir> --out <clean-dir>
```

Attribute FUNCAGUA for its supplied files. The CHIRPS monthly dataset retains `value_center`, the
series used by both comparison workbooks; `mean`, `min` and `max` are not ingested until their
definitions are documented. The long monthly FUNCAGUA values are independent micro-watershed means,
not monthly accumulations, and are not summed across months.

## Add a connector

1. Write tests first, with a small recorded or synthetic fixture. CI never calls external APIs.
2. Implement the `Connector` protocol in `agrofarm_data/connector.py` (`dataset`, `version`, `fetch`,
   `clean`, `write_clean`). `tests/fakes.py` has a small example.
3. `write_clean` writes the Parquet or GeoTIFF output and the manifest, and returns the manifest path.
4. Run it with `run_job(connector, raw_dir, out_dir)`: it stores the raw files once, cleans from the stored
   copies and validates the manifest. Running it again changes nothing.
5. Add the connector module to the image smoke test in `.github/workflows/ci.yml`.

The five connectors and `terrain/` predate the protocol. They have their own `write_clean` and are checked
against the manifest schema in `tests/test_existing_manifests.py`; they have not been moved onto `Connector` yet.
