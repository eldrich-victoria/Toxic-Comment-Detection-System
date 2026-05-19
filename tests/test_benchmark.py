import pytest
from benchmarking.metrics_engine import MetricsEngine

def test_metrics_engine():
    predictions = [
        {"prediction": 0.9, "ground_truth": 1, "latency_ms": 10},
        {"prediction": 0.1, "ground_truth": 0, "latency_ms": 15}
    ]
    metrics = MetricsEngine.calculate_metrics(predictions)
    assert metrics["accuracy"] == 1.0
    assert metrics["avg_latency"] == 12.5
