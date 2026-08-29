import os

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
