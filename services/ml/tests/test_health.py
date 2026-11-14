from fastapi.testclient import TestClient

from agrofarm_ml.main import create_app


def test_healthz_reports_alive() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
