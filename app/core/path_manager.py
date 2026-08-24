from pathlib import Path


class PathManager:

    # =====================================================
    # BASE DIRECTORY
    # =====================================================

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # =====================================================
    # CORE DIRECTORIES
    # =====================================================

    APP_DIR = BASE_DIR / "app"

    CONFIG_DIR = BASE_DIR / "configs"

    DATASETS_DIR = BASE_DIR / "datasets"

    DATABASE_DIR = BASE_DIR / "database"

    BENCHMARKING_DIR = BASE_DIR / "benchmarking"

    TESTS_DIR = BASE_DIR / "tests"

    TEMP_UPLOADS_DIR = BASE_DIR / "temp_uploads"

    # =====================================================
    # RESULTS DIRECTORIES
    # =====================================================

    BENCHMARK_RESULTS_DIR = BASE_DIR / "benchmark_results"

    CSV_RESULTS_DIR = BENCHMARK_RESULTS_DIR / "csv"

    JSON_RESULTS_DIR = BENCHMARK_RESULTS_DIR / "json"

    REPORTS_DIR = BENCHMARK_RESULTS_DIR / "reports"

    CHARTS_DIR = BENCHMARK_RESULTS_DIR / "charts"

    LOGS_DIR = BENCHMARK_RESULTS_DIR / "logs"

    RAW_PREDICTIONS_DIR = BENCHMARK_RESULTS_DIR / "raw_predictions"

    # =====================================================
    # MODEL DIRECTORIES
    # =====================================================

    VERSION_1_MODELS_DIR = BASE_DIR / "version_1" / "models"

    VERSION_2_MODELS_DIR = BASE_DIR / "version_2" / "models"

    # =====================================================
    # DATABASE FILES
    # =====================================================

    SQLITE_DB_PATH = DATABASE_DIR / "toxic_comments_benchmark.db"

    SCHEMA_PATH = DATABASE_DIR / "schema.sql"

    # =====================================================
    # CONFIG FILES
    # =====================================================

    MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"

    # =====================================================
    # CREATE REQUIRED DIRECTORIES
    # =====================================================

    REQUIRED_DIRS = [
        CONFIG_DIR,
        DATASETS_DIR,
        DATABASE_DIR,
        BENCHMARK_RESULTS_DIR,
        CSV_RESULTS_DIR,
        JSON_RESULTS_DIR,
        REPORTS_DIR,
        CHARTS_DIR,
        LOGS_DIR,
        RAW_PREDICTIONS_DIR,
        TEMP_UPLOADS_DIR,
    ]


# =====================================================
# AUTO-CREATE DIRECTORIES
# =====================================================

for directory in PathManager.REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)