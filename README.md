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

## Start here

- **Rules for people and agents:** [`AGENTS.md`](AGENTS.md). Read it before changing anything.
- **Tasks:** the GitHub issues of this repository. Issues labeled `agent-ready` can be given to an agent. Issues labeled `needs-human` need a person.
- **Shared plan and dataset catalog:** the `agrofarm-docs` repository.

## The short version of the rules

1. Everything is written in English.
2. Tests first (TDD). CI must be green.
3. Branch from `develop`, open a pull request into `develop`. Nobody pushes to `develop` or `main` directly.
4. `develop` deploys to staging. `main` deploys to production after manual approval.
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
