-- SQLite Schema for Toxic Comment Benchmarking & Evaluation Framework

PRAGMA foreign_keys = ON;

-- Benchmark Runs Table
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    status TEXT NOT NULL CHECK(status IN ('STARTED', 'COMPLETED', 'FAILED'))
);
CREATE INDEX idx_benchmark_runs_status ON benchmark_runs(status);

-- Model Predictions Table
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    sample_id TEXT NOT NULL,
    text_input TEXT NOT NULL,
    normalized_text TEXT,
    ground_truth INTEGER,
    prediction REAL NOT NULL,
    confidence REAL,
    latency_ms REAL,
    xai_placeholder TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);
CREATE INDEX idx_model_predictions_run_id ON model_predictions(run_id);
CREATE INDEX idx_model_predictions_model_name ON model_predictions(model_name);

-- Model Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);
CREATE INDEX idx_model_metrics_run_id ON model_metrics(run_id);

-- XAI Outputs Table
CREATE TABLE IF NOT EXISTS xai_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    xai_method TEXT NOT NULL,
    feature_importance TEXT NOT NULL, -- JSON formatted
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(prediction_id) REFERENCES model_predictions(id) ON DELETE CASCADE
);

-- Fairness Results Table
CREATE TABLE IF NOT EXISTS fairness_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    demographic_group TEXT NOT NULL,
    disparate_impact_ratio REAL,
    equal_opportunity_diff REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);
CREATE INDEX idx_fairness_results_run_id ON fairness_results(run_id);

-- Drift Results Table
CREATE TABLE IF NOT EXISTS drift_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    drift_score REAL NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);

-- Ensemble Results Table
CREATE TABLE IF NOT EXISTS ensemble_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ensemble_method TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    final_prediction REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
);

-- Error Analysis Table
CREATE TABLE IF NOT EXISTS error_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    analysis_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(prediction_id) REFERENCES model_predictions(id) ON DELETE CASCADE
);
