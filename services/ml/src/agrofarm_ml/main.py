"""Application factory. Run with: uvicorn agrofarm_ml.main:create_app --factory"""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from agrofarm_ml import stub
from agrofarm_ml.contract import load_contract
from agrofarm_ml.leaves.model import MAX_BYTES, LeafModel
from agrofarm_ml.rotations import engine
from agrofarm_ml.schemas import RotationScoreRequest, SimulateRequest


def create_app(leaf_model: LeafModel | None = None) -> FastAPI:
    app = FastAPI(title="AgroFarm ml-service", version="0.1.0", docs_url=None, redoc_url=None)
    # The contract file is the source of truth: publish it instead of a generated schema.
    app.openapi = load_contract  # type: ignore[method-assign]
    leaf_directory = os.environ.get("LEAF_MODEL_DIR")
    if leaf_model is None and leaf_directory:
        leaf_model = LeafModel.load(Path(leaf_directory))

    @app.post("/v1/leaves/assess")
    async def assess_leaf(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";")[0].strip().lower() not in {
            "image/jpeg",
            "image/png",
        }:
            raise HTTPException(415, "Upload JPEG or PNG image bytes")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "Image exceeds the 5 MiB limit")
        if leaf_model is None:
            raise HTTPException(503, "No trained leaf classifier is configured")
        try:
            # CPU work must not block the async request loop.
            from starlette.concurrency import run_in_threadpool

            return await run_in_threadpool(leaf_model.assess, bytes(data))
        except OverflowError as exc:
            raise HTTPException(413, "Image exceeds the decoded pixel limit") from exc
        except ValueError as exc:
            raise HTTPException(400, "Image could not be assessed") from exc

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
