# Cloud Run Rollback Runbook (`ml-service` and `data-service`)

This document describes how to roll back `ml-service` (Cloud Run Service, internal IAM-only) and `data-service` (Cloud Run Job) in `staging` or `production`.

## 1. Security and Invocation Policy

- `ml-service-<env>` is deployed with `--no-allow-unauthenticated` and is only invokable by `sa-core-api-<env>` via IAM OIDC tokens.
- `data-service-<env>` runs as an on-demand Cloud Run Job under `sa-data-service-<env>`.
- All CI/CD authentication uses Workload Identity Federation via encrypted GitHub Environment Secrets (`GCP_WIF_PROVIDER`, `GCP_SA_CI_DEPLOYER`, `GCP_PROJECT_ID`, `GCP_REGION`, `GAR_REPOSITORY`).

---

## 2. Rolling Back `ml-service` (Cloud Run Service)

```bash
ENV="staging" # or "prod"
gcloud run revisions list \
  --service="ml-service-${ENV}" \
  --region="${GCP_REGION}" \
  --project="${GCP_PROJECT_ID}" \
  --limit=5

PREVIOUS_REVISION="ml-service-staging-00001-xyz"
gcloud run services update-traffic "ml-service-${ENV}" \
  --to-revisions="${PREVIOUS_REVISION}=100" \
  --region="${GCP_REGION}" \
  --project="${GCP_PROJECT_ID}"
```

---

## 3. Rolling Back `data-service` (Cloud Run Job)

Pin `data-service-<env>` back to the previous commit SHA image in Artifact Registry:

```bash
ENV="staging" # or "prod"
PREVIOUS_SHA="<previous-commit-sha>"
gcloud run jobs update "data-service-${ENV}" \
  --image="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GAR_REPOSITORY}/data-service:${PREVIOUS_SHA}" \
  --region="${GCP_REGION}" \
  --project="${GCP_PROJECT_ID}"
```
