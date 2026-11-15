"""Application factory. Run with: uvicorn agrofarm_ml.main:create_app --factory"""

from typing import Any

from fastapi import FastAPI

from agrofarm_ml import stub
from agrofarm_ml.contract import load_contract
from agrofarm_ml.rotations import engine
from agrofarm_ml.schemas import RotationScoreRequest, SimulateRequest


def create_app() -> FastAPI:
    app = FastAPI(title="AgroFarm ml-service", version="0.1.0", docs_url=None, redoc_url=None)
    # The contract file is the source of truth: publish it instead of a generated schema.
    app.openapi = load_contract  # type: ignore[method-assign]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"items": [*stub.MODELS, engine.model_card()]}

    @app.post("/v1/simulate")
    def simulate(request: SimulateRequest) -> dict[str, Any]:
        return stub.simulate(request)

    @app.post("/v1/rotations/score")
    def score_rotations(request: RotationScoreRequest) -> dict[str, Any]:
        strategies = engine.score(
            request.current_rotation,
            request.priorities.weights,
            request.priorities.constraints,
            request.scenario.model_dump(),
        )
        return {
            "disclaimer": stub.DISCLAIMER,
            "strategies": strategies,
            "provenance": {**engine.provenance(), "generated_at": stub.now_iso()},
        }

    return app
