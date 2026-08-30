from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.predictor import Predictor
from app.db.database import Base, engine

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/image_quality_rf.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    predictor = Predictor(MODEL_PATH)
    predictor.load()
    app.state.predictor = predictor
    yield


app = FastAPI(
    title="AI Image Quality Assessment",
    version="1.0.0",
    description="Hybrid engineered-feature and classical-ML image quality analysis.",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/scoring", include_in_schema=False)
@app.get("/scoring/", include_in_schema=False)
def scoring_guide() -> FileResponse:
    return FileResponse(FRONTEND / "scoring.html")
