"""Utilities for the benchmarking pipeline."""
from typing import List, Dict, Any
import asyncio
from app.core.logger import logger
from app.core.exceptions import BenchmarkSystemError

class AsyncBenchmarkRunner:
    """Async-ready runner for benchmarking models."""
    
    def __init__(self, models: List[str], config: Dict[str, Any]):
        self.models = models
        self.config = config
        
    async def run_model_eval(self, model_name: str) -> Dict[str, Any]:
        """Simulate async model evaluation."""
        logger.info(f"Starting evaluation for model: {model_name}")
        await asyncio.sleep(0.1) # Simulate async workload
        logger.info(f"Completed evaluation for model: {model_name}")
        return {"model": model_name, "status": "success", "metrics": {"accuracy": 0.95}}

    async def execute_all(self) -> List[Dict[str, Any]]:
        """Run evaluations concurrently."""
        try:
            tasks = [self.run_model_eval(model) for model in self.models]
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            logger.error(f"Error during async benchmarking: {e}")
            raise BenchmarkSystemError(f"Execution failed: {e}") from e
