"""Application factory. Run with: uvicorn agrofarm_ml.main:create_app --factory"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="AgroFarm ml-service", version="0.1.0")

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
