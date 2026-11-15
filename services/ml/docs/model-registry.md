# Model registry (ML-050)

Every answer of `ml-service` names the model version that produced it (`provenance`). The registry
says where a model lives, what its card must contain, and what happens when a deploy has no valid
model.

## Layout

One directory per model and version, with the same tree everywhere:

```
models/<name>/<version>/
  registry_card.json    required; validated by agrofarm_ml.registry
  serving_card.json     the artifact's own card (preferred) ...
  model_card.json       ... or the card the training code wrote
  <model files>         read and checksummed by the model's own load()
```

| Where | Path |
|---|---|
| In the repository (cards only) | `services/ml/registry/<name>/<version>/registry_card.json` |
| In the image | `/app/runtime-models/<name>/<version>/` (files from the checksum-pinned release, cards copied from the repository) |
| In Cloud Storage (operator, see below) | `gs://<models bucket>/models/<name>/<version>/` |

Models today: `maize-leaf/0.1.1` and `ndvi-outlook/0.1.0`.

## The card

`registry_card.json` must have a value for each of: `name`, `version`, `model_version`, `status`,
`purpose`, `label` (what it predicts, or the proxy), `training_data_versions`, `features`,
`held_out_metrics` (metrics on years or groups the model did not see) and `limits`. The name and
version must equal the directory. Extra fields (license, review status, acceptance) are allowed.

The cards are copied from the experiment cards (`experiments/...`); a test checks that the numbers
agree, so a card cannot drift from the measured result. The vegetation card keeps the failed
verdict (`no_skill_over_climatology`) next to the owner's acceptance for limited use.

## Refusing to start

A deploy that needs models sets `REQUIRE_MODELS` to a list of `name/version`, separated by `;` or
`,` (the CD uses `;` because `gcloud` splits environment variables on commas). At start-up the
service checks each model: present, complete card, and an artifact card whose `model_version` is
the one the registry card names. If any check fails, `create_app()` raises `RegistryError` and the
service does not start, so a Cloud Run revision without a valid model never takes traffic and the
CD step fails instead.

`MODEL_REGISTRY_DIR` (default `/app/runtime-models`) is where the models are looked for. With
`REQUIRE_MODELS` empty (local runs, tests) nothing is checked and a model that is not configured
keeps answering 503 on its routes.

## Adding a model

1. Write its card in `services/ml/registry/<name>/<version>/registry_card.json`.
2. Package its files into `runtime-models/<name>/<version>/` (see `deploy/prepare_*.py`).
3. Add `<name>/<version>` to `REQUIRE_MODELS` in `.github/workflows/cd.yml` (staging and production).

## Publishing to Cloud Storage (not done, not verified)

Cloud Storage is not yet the source of the deployed models: the image is. To keep a copy under the
layout above, someone with write access to the models bucket can run, from `services/ml`:

```
gcloud storage rsync --recursive runtime-models gs://<models bucket>/models
```

The bucket name is `GCS_BUCKET_MODELS` in the Terraform of `agrofarm-api`. Nothing reads this copy
and these commands were not run. Loading models from the bucket at start-up would need a read
permission for the `ml-service` account and is not implemented.
