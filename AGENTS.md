# AGENTS.md: Working Rules for `agrofarm-analytic` (services `ml-service` and `data-service`)

**Everything in this repository is written in English. This is the highest priority rule and it overrides any other instruction.** See rule G1.

This repository holds two internal services: `ml-service` (section 7) and `data-service` (section 8). The rules for `core-api` live in `agrofarm-api`.

---

## 0. How to use this document

- Agents: read Part A (global rules) and Part B (backend rules) fully, then the sections for your service, then only the issue you were given.
- Humans: send tasks as GitHub issues using the template in Part F. One task, one branch, one pull request.
- If a rule conflicts with a task description, stop and ask. Do not guess.

## 1. Product context

We are building a simulator that predicts how crops will grow under given weather, soil and global climate conditions (for example El Nino and La Nina), and recommends crop rotation actions after a farmer uploads data about a specific farm. The UI is a React app with interactive maps (see `AGENTS.md` in `agrofarm-frontend`).

The backend must:

1. Let people register and log in securely.
2. Store farms, uploaded farm data, scenarios and recommendations.
3. Serve map layers and climate time series.
4. Run simulations with uncertainty ranges.
5. Compare **at least three rotation strategies, including the current one**, and return for each: why it was suggested, evidence, expected benefits, information gaps and risks that need local review.
6. Support search by name using trigrams.
7. Keep full provenance: which data versions and which model version produced each answer.

The tool **supports** decisions. It never replaces farmers or local agronomists. Responses must never present predictions as certain.

---

# PART A: Global rules (identical in every repository)

The canonical copy of Part A lives in `agrofarm-docs/AGENTS.md`. Change it there first, then copy the same text to the other three repositories the same day.

## G1. Language: English first

- All code identifiers, comments, commit messages, branch names, PR titles and descriptions, issues, documentation, API fields, dataset and column names, log messages, error messages and **UI copy** are in English.
- Spanish exists only as an i18n catalog in `agrofarm-frontend`, added in the final phase. It never appears in source code, tests, commits or docs.
- Team source documents written in Spanish stay outside the repositories. Their English versions live in `agrofarm-docs`.
- Chatting with humans in Spanish is fine. Nothing produced in a repo is in Spanish.

## G2. Test-driven development (mandatory)

1. **Red:** write a failing test for the behavior. Run it. Confirm it fails for the right reason.
2. **Green:** write the smallest code that makes it pass.
3. **Refactor:** clean up with all tests green.

- Bug fix: first write a regression test that reproduces the bug.
- Prefer separate commits: `test: ...` first, then `feat: ...` or `fix: ...`.
- Never delete, skip or weaken a test to make a change pass. If a test is wrong, explain why in the PR.
- CI enforces at least 80% line coverage on changed code.

## G3. CI/CD (mandatory)

- GitHub Actions. Every pull request runs: lint, type check, unit tests, integration tests, Docker image build, vulnerability scan.
- **A pull request merges only when every check is green.** The single required status check is the job named `ci-gate`. It depends on every other job and fails if any of them failed or was cancelled. Keep that job name stable. New jobs are added to its `needs` list.
- `develop` and `main` are always green and always deployable.
- Merge to `develop` deploys to **staging** automatically. Merge to `main` deploys to **production** after a manual approval.
- No manual deploys. The pipeline is code in the repo. Rollback means redeploying the previous image tag.
- Authenticate to GCP from CI with Workload Identity Federation. Never store service account JSON keys.

### G3.1 Test scope: full suite first, touched modules when it gets slow

- Default: every pull request runs the **full** test suite of the repo.
- When the pull request pipeline gets too slow (median above 10 minutes over the last 20 runs), switch pull requests to **affected-only** mode: run only the tests of the modules the change touches.
  - Detect touched modules with path filters in the workflow.
  - Inside a module, select tests by changed files (for example `vitest --changed origin/develop`, or one pytest path per package).
- Safeguards that always apply in affected-only mode:
  1. A change to shared files runs everything: lockfiles, CI workflows, Dockerfiles, root configuration, the OpenAPI contract, database migrations, shared test setup.
  2. The full suite always runs on every push to `develop`, on every release pull request into `main`, and nightly.
  3. `ci-gate` treats a skipped job as passed and a failed or cancelled job as failed, so path filters never block or bypass the required check.
- Turning on affected-only mode is its own pull request and states the measured timings.
- Slow tests are never fixed by deleting, skipping or weakening tests (rule G2).

## G4. Docker (mandatory)

- Every repo that ships a service has a multi-stage `Dockerfile` (pinned base image versions, non-root user, healthcheck) and a `docker-compose.yml` for local development.
- `docker compose up` must give a working local environment.
- Configuration comes from environment variables. Commit `.env.example`. Never commit `.env`.
- Standard Makefile targets in every repo: `make dev`, `make test`, `make lint`, `make typecheck`, `make build`, `make check` (lint + typecheck + test).

## G5. Microservices

Four repositories. Every service is deployable alone.

| Repo | Service(s) | Responsibility |
|---|---|---|
| `agrofarm-frontend` | `frontend` | React + Vite single page app (maps, simulation UI) |
| `agrofarm-api` | `core-api` | Public REST API: auth, farms, uploads, search, orchestration. Owns PostgreSQL. |
| `agrofarm-analytic` | `ml-service` | Internal service: crop simulation model, uncertainty, rotation scoring |
| `agrofarm-analytic` | `data-service` | Data pipelines: ingestion, cleaning, harmonization of weather, soil and satellite data |
| `agrofarm-docs` | none | Team plan, dataset catalog, ADRs, operations runbook, decisions and non-code tasks |

- Services talk only through HTTP contracts defined in OpenAPI. No shared code between services, including the two services inside `agrofarm-analytic`.
- No reaching into another service's storage. The one published data contract is the `clean/` datasets that `data-service` writes with a manifest: other services read them read-only, by manifest version.
- Only `core-api` talks to PostgreSQL.
- Contract first: change the OpenAPI file, then tests, then code.

## G6. Cloud: GCP

Cloud Run (services and jobs), Cloud SQL for PostgreSQL, Artifact Registry, Secret Manager, Cloud Storage. Document every manual cloud step in `docs/ops.md` of `agrofarm-docs`.

## G7. Security baseline

- No secrets in git, logs, tests or screenshots. Use Secret Manager and env vars.
- Validate all input at the boundary. Never trust the client.
- Least privilege for every credential and service account.
- Dependencies are pinned by lockfile and scanned in CI.

## G8. Git workflow

- Two long-lived branches: `develop` (integration, default branch, deploys to staging) and `main` (production releases).
- Work happens on short-lived branches cut from `develop`: `feat/<task-id>-<slug>`, `fix/...`, `chore/...`, `docs/...`, `test/...`, `refactor/...`, `ci/...`.
- **Every change reaches `develop` through a pull request. Pull requests from work branches target `develop` only, never `main`.**
- `main` only receives release pull requests from `develop`. An urgent production fix uses a `hotfix/...` branch cut from `main`, with a pull request into `main` and a second one back into `develop`.
- No direct pushes and no force pushes to `develop` or `main`.
- Conventional Commits. Squash merge into `develop`. Release pull requests into `main` use a merge commit so the two branches do not diverge.
- Small pull requests (aim for under 400 changed lines).
- A human reviews and merges. **Agents never push to `develop` or `main` and never merge their own PR.**

## G9. Rules for agents

1. Read this file and the README first.
2. Take only issues labeled `agent-ready`. Issues labeled `needs-human` are decisions or account work for people.
3. One task per branch and per PR. Stay inside the scope of the task. The PR targets `develop` and says `Closes #<issue>`.
4. Ambiguous requirement: ask. Do not guess silently.
5. No new dependency without a one-line justification in the PR description.
6. Run `make check` before saying the work is done.
7. Never run destructive commands (force push, dropping databases, deleting outside the workspace).
8. Update docs and the OpenAPI file together with the code.
9. In the final message report: what was done, what was not done, what you could not verify.

## G10. Domain safety

Recommendations are decision support. Every recommendation, in the API and in the UI, must carry reasons, evidence, data gaps, risks and uncertainty. Never present model output as certain. Never remove the disclaimer.

## G11. AI use traceability

The NASA Space Apps submission rules require that we say where and how AI tools were used, in descriptive text and in metadata. Every agent and every person who uses an AI tool leaves this trace. No trace, no merge.

1. **Commit metadata.** Every commit written with AI help ends with one trailer per tool:
   `Assisted-by: <tool> (<model id>)`, for example `Assisted-by: Claude Code (claude-fable-5-2)`.
   A commit written without AI has no trailer. Never add the trailer to work the tool did not touch, and never leave it out.
2. **Pull request text.** Every pull request fills the "AI use" section of the template: the tool and model, what it was used for (code, tests, docs, review, data analysis, translation), what it was **not** used for, and which person reviewed the output. "None" is a valid answer only if it is true. A squash merge keeps the `Assisted-by` trailers in the final commit message.
3. **Repository text.** `AI_USAGE.md` at the root of each repo is the public statement: the tools the team used in this repo, for what, and how to list the AI-assisted commits and pull requests. The README links to it. Whoever uses a tool that is not listed there adds it in the same pull request.
4. **Generated files.** A file that is mostly AI output and is not source code (a dataset, a table of crop traits, a long text) says so in its first lines or in its manifest (`generated_by`, `reviewed_by`).
5. **Media.** AI-generated images and videos carry a visible watermark ("AI-generated"). AI-generated audio is acknowledged in the text next to it and in the file metadata. No NASA logo, flag or mission identifier in generated content.
6. **Limits.** AI tools speed up the work. They do not choose the idea of the solution, and a person on the team reviews everything that is merged (rule G8). Never paste secrets, credentials or personal data into an AI tool.
7. **Machine learning inside the product** is not covered by this rule. It is documented by the model card and the provenance fields (rule G10).

## G12. Public access for judges

- Every link in the submission opens in a private browser window with no password, no permission request and no registration. This includes the repositories, the demo and the live application.
- The live application has an entry that works without login (a public demo mode with example data). Login stays available for real farm data and is never the only way in.
- Credentials, including demo credentials, are never written in documentation, slides or the submission form.

---

# PART B: Backend rules (all backend services)

## 2. Stack

Python for all three services so the team shares one language.

| Concern | Choice |
|---|---|
| Web framework | FastAPI with Pydantic v2 |
| Lint and format | Ruff |
| Types | mypy (strict on new code) |
| Tests | pytest, pytest-cov, httpx test client |
| Dependency management | `uv` or `pip-tools` with a locked file (pick one per repo and record it in the README) |
| Container | Python slim image, multi-stage, non-root |

Use current stable versions at project start. Anything outside this table needs a justification in the PR.

## 3. Contracts between services

- Every service publishes `openapi.yaml` in `docs/contract/` and serves it at `/openapi.json`. Contract first.
- Public paths are versioned: `/v1/...`. A breaking change means a new version and a short ADR in `docs/adr/`.
- Errors use `application/problem+json` (RFC 7807) with a stable `code` field.
- Lists are paginated (`limit`, `cursor`). Writes that can be retried accept an `Idempotency-Key`.
- Propagate `X-Request-Id` on every call and put it in every log line.
- Health: `GET /healthz` (process is alive) and `GET /readyz` (dependencies reachable).
- Service to service calls on Cloud Run use Google-signed identity tokens. `ml-service` and `data-service` are **not** publicly reachable (`--no-allow-unauthenticated`).
- Contract tests: schemathesis against the running service in CI. The frontend generates its TypeScript types from the `core-api` contract.
- **Contract stubs first:** within the first days `core-api` publishes a stub contract and stub endpoints with realistic example data, so the frontend and `ml-service` are not blocked.

## 4. Layered architecture

```
Routers (HTTP only)  ->  Services (use cases)  ->  Repositories (data access)  ->  Database / storage
                              |
                          Domain models and pure functions
```

```
src/<package>/
  api/            routers, request and response schemas, dependencies
  middleware/     request id, logging, CORS, rate limit, auth, error handler
  services/       business logic, no HTTP and no SQL
  repositories/   SQL and storage access, no business logic
  domain/         pure models and functions
  clients/        HTTP clients for other services (timeouts, retries, circuit breaker)
  config.py       settings from environment, validated at startup
tests/
  unit/ integration/ contract/ e2e/
```

Rules:

1. Routers contain no business logic and no SQL.
2. Services never import FastAPI. Repositories never import services.
3. Dependency injection for repositories, clients and settings, so tests can replace them.
4. Middleware order: request id, structured logging, CORS (explicit allowlist, never `*` with credentials), rate limit, authentication, error handler.
5. Every outbound HTTP call has a timeout, a bounded retry for idempotent calls, and a defined fallback or error.
6. Settings are validated at startup. The app refuses to start with missing secrets.

## 5. Testing rules (TDD applied)

| Level | What | Notes |
|---|---|---|
| Unit | Services, domain, pure functions | No network, no database |
| Integration | Repositories, migrations, routers | **Real PostgreSQL** in a container (Docker Compose service or Testcontainers). No SQLite substitutes. |
| Contract | OpenAPI conformance | schemathesis in CI |
| Security | Service identity | Calls without a valid Google identity token are rejected |
| End to end | Full flow across compose | Smoke level, runs in CI |

- Use factories and fixtures. Freeze time in tests that depend on it.
- External APIs are never called in CI. Use recorded fixtures.
- Every endpoint has tests for success, validation error, unauthenticated, and forbidden (another user's resource).

## 6. Repository layout: two services in one repo

```
services/
  ml/      ml-service    pyproject + lockfile, src/agrofarm_ml/, tests/, Dockerfile, docs/contract/openapi.yaml
  data/    data-service  pyproject + lockfile, src/agrofarm_data/, tests/, Dockerfile, docs/contract/openapi.yaml
Makefile            make check runs both. make check-ml and make check-data run one.
docker-compose.yml  both services, with core-api replaced by a stub
.github/workflows/  one job group per service
```

Rules:

1. The two services are independent deployables. Each has its own package, lockfile, image, contract and tests. The layout of section 4 applies inside each `services/<name>/`.
2. No import crosses between `services/ml` and `services/data`. A test enforces it.
3. `ml-service` reads the `clean/` datasets of `data-service` read-only and by manifest version. It never writes there and never reads `raw/`.
4. CI is always scoped per service with path filters: a change under `services/ml/` runs the ml jobs, a change under `services/data/` runs the data jobs, and a change to root files runs both. This is separate from rule G3.1, which applies inside each service.
5. A pull request touches one service. A change that needs both is two pull requests, contract first.

## 7. `ml-service`

Internal service. Stateless. Serves the crop simulation model and the rotation engine.

### 7.1 API

- `POST /v1/simulate`: inputs are location or farm features, a time range, and scenario deltas (rainfall percent, temperature delta, ENSO phase). Output: prediction, **quantile range**, top feature attributions, `model_version`, `data_versions`.
- `POST /v1/rotations/score`: inputs are candidate rotations, farm features and farmer priorities. Output per strategy: score breakdown, `why`, `evidence`, `benefits`, `gaps`, `risks`, `needs_local_review`.
- `GET /v1/models`: loaded models and their cards.

### 7.2 Modeling rules

- Training code is a Python package with a CLI (`src/<package>/training`). **Notebooks, including Google Colab, are allowed only for exploration.** Anything that ships lives in the package with tests. Notebooks are never imported by code and have outputs stripped (`nbstripout`).
- Fix random seeds. Same input and same model version give the same output.
- **Split validation by whole years, never randomly.** Hold out at least one El Nino year and one La Nina year.
- Start with a gradient boosting baseline (LightGBM or XGBoost). Add uncertainty with quantile regression. Explain with SHAP. A model more complex than the baseline must beat it on the held-out years.
- Missing values are `NaN`. Never use sentinel values such as `-99.9` (see data rules in section 8).
- Data validation on every input table (for example with `pandera`).
- The rotation engine is rule based and data driven. Crop traits (water use, nitrogen fixing, root depth, cycle length, soil cover) live in versioned tables, not in code. Rules are unit tested.
- Never trust the model outside its training range. Detect out-of-range inputs and flag them in the response.
- Labels: yield is not available per parcel from public sources. Document which label or proxy each model uses in its model card.

### 7.3 Artifacts and model cards

- Model artifacts are stored in Cloud Storage under `models/<name>/<version>/`.
- Each model has a model card (markdown or JSON): purpose, training data versions, features, metrics on held-out years (including El Nino and La Nina years), known limits.
- Models are loaded at startup. The service refuses to start without a valid model.

### 7.4 Tests

- Unit tests for feature engineering and the rotation rules.
- Golden tests on a tiny fixture dataset.
- Behavior tests: output has no `NaN`, quantiles are ordered, extreme scenario inputs do not crash.
- CI gate: metrics on the fixture are not worse than the stored baseline. Full training never runs in CI.

## 8. `data-service`

Internal. Runs as Cloud Run jobs triggered by Cloud Scheduler or manually. Produces datasets for the other services.

### 8.1 Connector rules

Each source is one connector with the same interface: `fetch -> validate -> normalize -> write -> register`.

| Source | Notes |
|---|---|
| INSIVUMEH (Excel) | Two sheets (1995-2022 and 2023-2025), header on row 5, **`-99.9` means missing**, data marked preliminary |
| NASA POWER | Daily climate, no key needed |
| CHIRPS | Rainfall |
| ERA5-Land | Hourly climate and soil, includes min and max through daily statistics |
| ENSO | RONI and ONI from NOAA CPC, monthly, mark each month El Nino, La Nina or neutral (threshold plus or minus 0.5 degrees C) |
| SMAP, MODIS, VIIRS, Sentinel-2 | Via AppEEARS or Planetary Computer. Confirm collection ids before coding. |
| SoilGrids, DEM | Static layers |
| Meteomatics | Budget 1,000 queries per day. Cache every response. |

- Normalization: ISO dates, EPSG:4326, SI-based units documented per dataset (rainfall mm per day, temperature degrees C, wind km/h), missing values as null.
- Keep raw data immutable in `raw/` and cleaned data in `clean/`. Outputs are Parquet (tables) and Cloud Optimized GeoTIFF (rasters) in Cloud Storage.
- Every output has a manifest (JSON): source, version, license and attribution, date range, spatial resolution, checksum, processing code version. Connectors register the manifest with `core-api` through an internal endpoint.
- Jobs are idempotent and resumable. Re-running does not duplicate data.
- Harmonization lives in one library module: resample to the common grid (0.05 degrees for training, finer for a farm), time aggregation to 10-day or monthly steps, anomaly computation against a per-cell climatology.
- Credentials (Earthdata, Meteomatics) come from Secret Manager. Respect rate limits and license terms.

### 8.2 Tests

- Connectors are tested with recorded fixtures. CI never calls external APIs.
- Test the normalization with the nasty cases: sentinel values, duplicate dates, missing days, wrong units, empty responses.
- Schema tests on every output table.

---

# PART C: Docker, CI/CD and GCP specifics

## 9. Docker

- Multi-stage build, slim base image with a pinned version, non-root user, `HEALTHCHECK`.
- `docker-compose.yml` in `core-api` runs the API and PostgreSQL (use an image that includes `pgvector` and, if approved, PostGIS).
- A full-stack compose file lives in `core-api` under `deploy/compose.full.yml`. It pulls the latest `develop` images of `frontend`, `ml-service` and `data-service` from Artifact Registry.
- Each other repo has its own compose with the other services replaced by stubs.

## 10. CI/CD pipeline (GitHub Actions)

On pull request:

1. Install with the lockfile.
2. Ruff, mypy.
3. Unit and integration tests with coverage gate (80% on changed code). Integration tests start PostgreSQL.
4. Migration test (empty database and upgrade path) for `core-api`.
5. Contract tests (schemathesis).
6. `pip-audit` and Trivy image scan.
7. Docker build.

All of these jobs are in the `needs` list of `ci-gate`. Test scope follows rule G3.1: full suite by default, touched modules only once the pipeline is slow.

On `develop` (after a pull request is merged):

1. Build and push the image to Artifact Registry (tag with the commit SHA).
2. Run database migrations as a Cloud Run job **before** deploying the new revision.
3. Deploy to Cloud Run staging. Smoke test.

On `main` (after a release pull request from `develop` is merged):

1. Wait for the manual approval on the GitHub environment `production`.
2. Run migrations, then deploy the image that was already built and tested on staging. Never rebuild for production.
3. Rollback by shifting traffic to the previous revision.

## 11. GCP rules

- One Cloud Run service or job set per service (`agrofarm-analytic` deploys two). Internal services use authenticated invocation only.
- Cloud SQL via the Cloud SQL connector. Secrets in Secret Manager mounted as env vars.
- Service accounts per service with minimum roles.
- Minimum instances 0 for cost. Set CPU and memory limits explicitly.
- Structured JSON logs to stdout (picked up by Cloud Logging). No secrets and no personal data in logs. Every line has `request_id`.
- Rasters and models in Cloud Storage with private buckets and uniform access.

---

# PART D: Phased plan and tasks

Each task is a GitHub issue with the same ID in its title. The full acceptance criteria live in the issue. `core` tasks are inside the cut line (minimum for the demo). `stretch` tasks come after it.

## Phase 0: Kickoff

- [ ] **AN-001** Scaffold the repo: two services, root Makefile, compose and per-service CI. *(core)*

## Phase 1: Data

- [ ] **AN-002** CD for both services: Cloud Run service for ml, Cloud Run jobs for data. *(core)*
- [ ] **ML-001** ml-service scaffold with deterministic stub `/v1/simulate` and `/v1/rotations/score`. *(core)*
- [ ] **ML-005** Decision note: which label or proxy the model predicts. *(core, needs-human)*
- [ ] **ML-038** Versioned crop traits table for the pilot crops. *(core)*
- [ ] **DS-001** data-service scaffold, connector interface and manifest schema. *(core)*
- [ ] **DS-010** INSIVUMEH connector (two sheets, `-99.9` to null). *(core)*
- [ ] **DS-015** Review the Rio Contreras rainfall data and decide whether to ingest it. *(stretch, needs-human)*
- [ ] **DS-020** NASA POWER connector (daily climate, including minimum and maximum temperature). *(core)*
- [ ] **DS-021** CHIRPS rainfall connector. *(core)*
- [ ] **DS-023** ENSO connector: RONI and ONI from NOAA CPC with a phase per month. *(core)*
- [ ] **DS-030** AppEEARS connector: SMAP soil moisture, MODIS and VIIRS vegetation series. *(stretch)*
- [ ] **DS-040** SoilGrids static layers. *(stretch)*
- [ ] **DS-045** Label datasets: GDHY, GGCP10 and SPAM 2010. *(core)*

## Phase 2: Baseline model and map

- [ ] **ML-010** Feature pipeline package with tests. *(core)*
- [ ] **ML-020** Baseline model with whole-year validation including El Nino and La Nina years. *(core)*
- [ ] **ML-040** Rotation engine: rule-based scoring of at least three strategies including the current one. *(core)*
- [ ] **DS-022** ERA5-Land daily connector. *(core)*
- [ ] **DS-025** Compare POWER and CHIRPS against INSIVUMEH (correlation and bias). *(stretch)*
- [ ] **DS-031** Planetary Computer connector: Sentinel-2 or HLS NDVI with cloud mask. *(stretch)*
- [ ] **DS-041** DEM terrain layers: slope, aspect, flow accumulation, erosion and waterlogging zones. *(stretch)*
- [ ] **DS-042** Crop mask from IO LULC or ESA WorldCover. *(stretch)*
- [ ] **DS-046** FAO GAEZ v5 suitability and attainable yield for the pilot crops. *(stretch)*
- [ ] **DS-050** Harmonization: common grid, 10-day and monthly steps, anomalies. *(stretch)*
- [ ] **DS-055** Climate indicators: dry spells, heavy rain days, hot days, cold nights, water balance. *(stretch)*
- [ ] **DS-056** Bare-field period detection from low NDVI. *(stretch)*
- [ ] **DS-070** Meteomatics point forecast with a disk cache and a daily budget. *(stretch)*

## Phase 3: Integration

- [ ] **ML-030** Quantile uncertainty and SHAP attributions in `/v1/simulate`. *(core)*
- [ ] **ML-050** Model registry layout in Cloud Storage and model cards. *(stretch)*
- [ ] **DS-060** Manifest registration with core-api and scheduled jobs. *(stretch)*
- [ ] **DS-080** ECOSTRESS water stress layer (optional). *(stretch)*

## Phase 4: Polish and demo

- [ ] **ML-060** Calibrate the model with farm uploads. *(stretch)*

---

# PART E: Definition of done

- [ ] Tests were written first and are in the commit history.
- [ ] `make check` passes locally and CI is green.
- [ ] Coverage on changed code is at least 80%.
- [ ] Everything written is English.
- [ ] OpenAPI, migrations, `.env.example` and docs are updated.
- [ ] No secrets, no personal data in logs.
- [ ] The PR targets `develop` and closes its issue.
- [ ] The PR description says what changed and what is not verified.

# PART F: Task instruction template (send this to an agent)

```
Task: <ID and title, e.g. BE-010 Auth endpoints>
Repo: agrofarm-analytic (service: <ml-service | data-service>)
Context: <why this matters, link to the section of AGENTS.md>
Scope: <directories and files allowed to change>
Out of scope: <what must not be touched>
Contract: <OpenAPI paths affected, update the contract first>
Acceptance criteria (write these as tests first):
  - ...
Constraints: English only. TDD. No new dependency without justification.
Deliver: branch cut from `develop`, pull request into `develop` with description, summary of `make check`,
         list of anything not verified.
```
