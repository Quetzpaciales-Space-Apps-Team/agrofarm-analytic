from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CD_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "cd.yml"
ROLLBACK_DOC_PATH = REPO_ROOT / "deploy" / "rollback.md"


def test_cd_workflow_exists_and_uses_wif_without_hardcoded_secrets() -> None:
    assert CD_WORKFLOW_PATH.is_file(), "Missing .github/workflows/cd.yml"
    raw = CD_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "branches:" in raw
    assert "- develop" in raw
    assert "- main" in raw
    assert "workflow_dispatch:" in raw

    # Workload Identity Federation via encrypted secrets, no JSON keys or hardcoded accounts
    assert "google-github-actions/auth@v2" in raw
    assert "secrets.GCP_WIF_PROVIDER" in raw
    assert "secrets.GCP_SA_CI_DEPLOYER" in raw
    assert "secrets.GCP_PROJECT_ID" in raw
    assert "credentials_json" not in raw
    assert ".iam.gserviceaccount.com" not in raw
    assert "agrofarm-spaceapps" not in raw


def test_cd_workflow_filters_changed_services_and_enforces_authenticated_ml_and_data_jobs() -> None:
    raw = CD_WORKFLOW_PATH.read_text(encoding="utf-8")

    # Path filter step detects changed services
    assert "services/ml/" in raw
    assert "services/data/" in raw

    # ml-service is deployed as Cloud Run service with authenticated invocation only
    assert "ml-service-staging" in raw
    assert "ml-service-prod" in raw
    # `gcloud run services update` rejects access flags ("unrecognized arguments"), so the
    # workflow must not pass one, and must never open the service to the public.
    assert "--no-allow-unauthenticated" not in raw
    assert "--allow-unauthenticated" not in raw
    assert "allUsers" not in raw.replace('grep -q "allUsers"', "")
    # After each ml-service deploy the workflow checks that the service is not public.
    assert raw.count("gcloud run services get-iam-policy ml-service-") == 2
    assert raw.count('grep -q "allUsers"') == 2

    # data-service is deployed as Cloud Run job
    assert "gcloud run jobs update data-service-staging" in raw
    assert "gcloud run jobs update data-service-prod" in raw

    # Environments for staging and manual-approval production
    assert "environment: staging" in raw
    assert "environment: production" in raw


def test_rollback_runbook_is_documented() -> None:
    assert ROLLBACK_DOC_PATH.is_file(), "Missing deploy/rollback.md"
    content = ROLLBACK_DOC_PATH.read_text(encoding="utf-8")
    assert "gcloud run services update-traffic" in content
    assert "ml-service" in content
    assert "data-service" in content
