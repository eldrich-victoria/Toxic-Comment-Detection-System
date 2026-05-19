"""Main entry point for Phase 1 Setup and Initialization."""
import asyncio
import sys
from app.core.path_manager import PathManager
from app.core.logger import logger
from app.core.config_loader import ConfigLoader
from app.core.constants import ConfigConstants
from benchmarking.benchmark_utils import AsyncBenchmarkRunner

def initialize_system():
    """Initialize system directories and load configs."""
    logger.info("Initializing Toxic Comment Benchmarking System Phase 1...")
    PathManager.initialize_directories()
    logger.info("Directories initialized successfully.")
    
    try:
        benchmark_cfg = ConfigLoader.load_yaml(ConfigConstants.BENCHMARK_CONFIG)
        logger.info(f"Loaded benchmark config: {benchmark_cfg.get('benchmark', {}).get('metrics')}")
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        sys.exit(1)

async def main():
    initialize_system()
    
    # Example async workflow validation
    runner = AsyncBenchmarkRunner(["baseline_model", "bert_model"], {})
    results = await runner.execute_all()
    logger.info(f"Test run completed: {results}")

if __name__ == "__main__":
    asyncio.run(main())
