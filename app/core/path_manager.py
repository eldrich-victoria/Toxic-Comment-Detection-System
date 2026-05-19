"""Path management utility using pathlib."""
from pathlib import Path
from typing import Final

class PathManager:
    """Centralized path management for the project."""
    
    ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
    
    APP_DIR: Final[Path] = ROOT_DIR / "app"
    CONFIG_DIR: Final[Path] = ROOT_DIR / "configs"
    DATABASE_DIR: Final[Path] = ROOT_DIR / "database"
    BENCHMARK_RESULTS_DIR: Final[Path] = ROOT_DIR / "benchmark_results"
    DATASETS_DIR: Final[Path] = ROOT_DIR / "datasets"
    LOGS_DIR: Final[Path] = ROOT_DIR / "logs"
    
    @classmethod
    def initialize_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        for path_attr in dir(cls):
            if path_attr.endswith("_DIR"):
                path = getattr(cls, path_attr)
                path.mkdir(parents=True, exist_ok=True)
