import os
from pathlib import Path

os.environ["DATABASE_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_empty_history():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert isinstance(health.json()["model_loaded"], bool)
        assert health.json()["status"] in {"ok", "degraded"}
        history = client.get("/api/history")
        assert history.status_code == 200
        assert history.json() == []


def test_unsupported_upload_type():
    with TestClient(app) as client:
        response = client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
        assert response.status_code == 415


def test_analysis_includes_plain_english_summary_when_model_exists():
    model = Path("models/image_quality_rf.joblib")
    sample = Path("samples/generated/clean.jpg")
    if not model.exists() or not sample.exists():
        return
    with TestClient(app) as client, sample.open("rb") as handle:
        response = client.post("/api/analyze", files={"file": (sample.name, handle, "image/jpeg")})
        assert response.status_code == 200
        assert len(response.json()["quality_summary"]) == 3
