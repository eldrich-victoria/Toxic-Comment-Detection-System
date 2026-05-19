"""Reusable system-wide constants."""

class ConfigConstants:
    """Constants related to configuration files."""
    LOGGING_CONFIG = "logging_config.yaml"
    BENCHMARK_CONFIG = "benchmark_config.yaml"
    MODEL_CONFIG = "model_config.yaml"
    FAIRNESS_CONFIG = "fairness_config.yaml"

class DBConstants:
    """Database-related constants."""
    SQLITE_DB_NAME = "toxic_comments_benchmark.db"
    SCHEMA_FILE = "schema.sql"

class BenchmarkConstants:
    """Benchmarking specific constants."""
    DEFAULT_BATCH_SIZE = 32
    SUPPORTED_METRICS = ["accuracy", "f1_score", "roc_auc", "precision", "recall"]
