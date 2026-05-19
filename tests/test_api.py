import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_benchmark_run_validation():
    # Should fail if missing required body
    response = client.post("/benchmark/run", json={})
    assert response.status_code == 422

def test_benchmark_run_success():
    payload = {
        "dataset_name": "test_data",
        "model_configs": [{"name": "test_model", "type": "sklearn"}]
    }
    response = client.post("/benchmark/run", json=payload)
    assert response.status_code == 200
    assert "run_id" in response.json()
