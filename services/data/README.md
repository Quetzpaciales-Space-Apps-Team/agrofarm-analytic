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
