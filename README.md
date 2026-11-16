# agrofarm-analytic

Two internal services: `ml-service` (crop simulation, uncertainty, rotation scoring) and `data-service` (ingestion and harmonization of weather, soil and satellite data).

Part of AgroFarm, a crop growth and rotation simulator for farmers. The tool supports decisions. It never replaces farmers or local agronomists.

## Run it

Needs Docker and [uv](https://docs.astral.sh/uv/) (dependency manager for both services, one `uv.lock` per service).

```
make check        # lint, typecheck and tests of both services
make check-ml     # only services/ml
make check-data   # only services/data
make dev          # ml-service on http://localhost:8081
docker compose run --rm data --help   # data-service jobs
```

Layout: `services/ml` is `ml-service` (package `agrofarm_ml`), `services/data` is `data-service` (package `agrofarm_data`). They never import each other. A test in each service enforces it. A pull request touches one service.

## What is in this repo

- `services/data` is `data-service`: connectors for external data, cleaning into clean tables, and a command line (`python -m agrofarm_data <command>`). See "Connectors".
- `services/ml` is `ml-service`: a rotation rule engine over a provisional crop traits table, a model registry, a rainfall baseline model, an experimental NDVI outlook model and a maize-leaf image screening model, served through a small API (`/v1/simulate`, `/v1/rotations/score`, `/v1/vegetation/simulate`, `/v1/leaves/assess`, `/v1/models`). See "Models".
- `experiments/` holds the records of two experiments from 2026-11-15: `2026-11-15-label-search` (the search for a yield label, no label beat climatology) and `2026-11-15-maize-leaf`.
- `deploy/rollback.md` describes rollback. CI and CD are in `.github/workflows/`.

## Connectors

Commands of `data-service` (run `docker compose run --rm data <command> --help` or `uv run python -m agrofarm_data <command> --help` in `services/data`):

| Command | Source |
|---|---|
| `insivumeh` | INSIVUMEH daily climate workbook (cleaning) |
| `rio-contreras` | reviewed Rio Contreras rainfall sources |
| `enso` | NOAA ONI and RONI text files |
| `nasa-power` | NASA POWER daily climate, from a CSV or with `--fetch point` / `--fetch regional` |
| `chirps` | CHIRPS rainfall windows, monthly dataset |
| `soilgrids` | SoilGrids layers cropped for an area |
| `worldcover`, `worldcover-polygon` | ESA WorldCover cropland mask and farm polygon report |
| `terrain` | DEM terrain layers |
| `planetary-ndvi` | NDVI with a cloud mask from Microsoft Planetary Computer |
| `meteomatics-forecast` | Meteomatics daily point forecast (cached, daily request budget; needs `METEOMATICS_USERNAME` and `METEOMATICS_PASSWORD`) |
| `appeears` | NASA AppEEARS polygon statistics (see below) |

### AppEEARS (`appeears`)

`--source` is one of `smap-l4` (SMAP L4 soil moisture), `modis-13q1` (MODIS vegetation) or `viirs-vnp13a1` (VIIRS vegetation), for a `--bbox WEST SOUTH EAST NORTH` and a `--start`/`--end` range.

- It needs NASA Earthdata credentials in the environment. Locally put them in a git-ignored `services/data/.env` as `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (names only here, see `.env.example`). AppEEARS rejects the Earthdata profile token with HTTP 403, so it needs a login with user name and password.
- The first run submits an AppEEARS task and the command exits with code 75 while the task is still running. Run the same command again to collect it. `--wait-seconds` keeps polling for a while before exiting with 75.
- `--min-usable-fraction` (vegetation only, default 0.5) is the share of usable pixels a composite needs to keep its values.
- Recorded test fixtures of the real API answers are in `services/data/tests/fixtures/appeears/`, so the tests need no network or credentials.

## Models

- **Rotation rule engine** (`agrofarm_ml.rotations`). Rule based, not a trained model. It scores the current rotation and candidate sequences from the crop traits table.
- **Crop traits table** (`services/ml/src/agrofarm_ml/rotations/data/crop_traits.json`, with `manifest.json`): version 0.1.0, status `provisional`. `values_verified_against_sources` is `false`: values were written by an AI assistant or recalled from FAO-56 and are not checked against the sources. The owner confirmed that a person with agronomy knowledge reviewed the table, but the reviewer identity, reviewed revision and corrections were not recorded. Treat every value as needing local agronomist review.
- **Model registry** (`services/ml/docs/model-registry.md`, PR #72). Cards are in `services/ml/registry/<name>/<version>/registry_card.json`; today `maize-leaf/0.1.1` and `ndvi-outlook/0.1.0`. The `REQUIRE_MODELS` environment variable lists the `name/version` pairs a deploy needs, separated by `;`; it is set in `.github/workflows/cd.yml`. If a required model is missing or its card is incomplete, the service refuses to start.
- **Rainfall baseline model** (see below). On the first run with real data it did not beat climatology.
- **NDVI outlook model (experimental, limited use).** It predicts a regional seasonal peak NDVI anomaly, not parcel yield and not an ENSO forecast. In leave-one-year-out evaluation it improved pooled quantile loss by 4.41% over climatology, below the 5% rule written before training, and its `no_skill_over_climatology` verdict stays on record. The project owner accepted limited use anyway (ADR 0005 in `agrofarm-docs`). Annual mean correlation was 0.127 and p10 to p90 coverage was 75.6% instead of 80%. Do not read it as a validated forecast. Details: `services/ml/ndvi-serving.md`.
- **Maize-leaf screening** classifies a photo into five classes; see `services/ml/docs/leaf-screening.md` for its source, split and limits.

## Rainfall baseline model

```
# 1. clean tables (data-service)
cd services/data
uv run python -m agrofarm_data insivumeh --input <INSIVUMEH.xlsx> --out ../../data/clean
uv run python -m agrofarm_data enso --oni <oni.ascii.txt> --roni <RONI.ascii.txt> --out ../../data/clean
# 2. train (ml-service)
cd ../ml
uv run python -m agrofarm_ml.training --climate ../../data/clean/climate_insivumeh_daily.parquet \
    --enso ../../data/clean/enso_monthly.parquet --out ../../data/models/rainfall/0.1.0
```

The model predicts the monthly rainfall total at the station with a p10 to p90 range, from the month of the year and the ENSO index (RONI) of the last six months. It is validated on whole held-out years, including El Nino and La Nina years, against the calendar-month climatology. The result is written to `model_card.json` with a `verdict` field. Read it before showing any number: on the first run with real data the model did **not** beat climatology.

## Operations notes

- On 2026-11-16 02:10 UTC the `ci` and `cd` workflows on `develop` failed with GitHub's annotation "The job was not started because recent account payments have failed or your spending limit needs to be increased". No job ran. The repository was private at the time. After it became public the runs were re-run and passed: `ci` succeeded; in `cd` the ml-service and data-service staging jobs deployed and production was skipped.
- Only the `main` and `develop` branches exist. Feature branches were deleted on 2026-11-15 after their pull requests merged.
- Pending, needs GCP access, NOT done: uploading the model artifacts to the staging models bucket (`gcloud storage rsync --recursive models-for-gcs gs://<PROJECT_ID>-models-staging/models`, where `<PROJECT_ID>` is the GitHub secret `GCP_PROJECT_ID`) and loading the Earthdata credentials into Secret Manager. Neither is confirmed.
- A push to `main` deploys to production Cloud Run with no required reviewers. Production has never been deployed before.

## Start here

- **Rules for people and agents:** [`AGENTS.md`](AGENTS.md). Read it before changing anything.
- **Tasks:** the GitHub issues of this repository. Issues labeled `agent-ready` can be given to an agent. Issues labeled `needs-human` need a person.
- **Shared plan and dataset catalog:** the `agrofarm-docs` repository.

## The short version of the rules

1. Everything is written in English.
2. Tests first (TDD). CI must be green.
3. Branch from `develop`, open a pull request into `develop`. Nobody pushes to `develop` or `main` directly.
4. `develop` deploys to staging. `main` deploys to production (see "Operations notes": no required reviewers are configured).
5. One task, one branch, one pull request.

## Repositories

| Repo | Service(s) |
|---|---|
| `agrofarm-frontend` | `frontend` |
| `agrofarm-api` | `core-api` |
| `agrofarm-analytic` | `ml-service`, `data-service` |
| `agrofarm-docs` | documentation only |

## AI assistance

Parts of this repository were written with AI coding assistants and reviewed by the team. See [`AI_USAGE.md`](AI_USAGE.md) for the tools, what they were used for and how to trace each use.
