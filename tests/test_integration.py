import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_full_pipeline_mock():
    # 1. Trigger run
    payload = {"dataset_name": "test_data", "model_configs": []}
    res = client.post("/benchmark/run", json=payload)
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    
    # 2. Check results endpoint
    res = client.get("/benchmark/results")
    assert res.status_code == 200
    
    # 3. Check reports
    res = client.get(f"/benchmark/report/{run_id}")
    assert res.status_code == 200
