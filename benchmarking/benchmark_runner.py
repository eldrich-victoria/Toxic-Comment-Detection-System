"""Benchmark runner for orchestrating evaluations."""
import asyncio
import json
import time
from typing import List, Dict, Any

from app.core.logger import logger
from benchmarking.inference_engine import InferenceEngine
from database.storage_manager import StorageManager

class BenchmarkRunner:
    """Orchestrates benchmark runs, coordinating inference and storage."""
    
    def __init__(self, inference_engine: InferenceEngine, storage_manager: StorageManager, config: Dict[str, Any]):
        self.engine = inference_engine
        self.storage = storage_manager
        self.config = config
        
    async def run_benchmark(self, dataset: List[Dict[str, Any]], model_configs: List[Dict[str, Any]]):
        """Execute a full benchmark run across discovered models."""
        run_name = f"benchmark_run_{int(time.time())}"
        logger.info(f"Starting benchmark run: {run_name}")
        
        # 1. Register Run
        config_snapshot = json.dumps(self.config)
        run_id = await self.storage.insert_benchmark_run(run_name, config_snapshot)
        
        try:
            # 2. Discover and Load Models
            self.engine.discover_and_load_models(model_configs)
            
            # 3. Process batches
            batch_size = self.config.get("benchmark", {}).get("default_batch_size", 32)
            batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
            
            all_predictions = []
            
            for model_cfg in model_configs:
                model_name = model_cfg["name"]
                logger.info(f"Running inference for model: {model_name}")
                
                tasks = [self.engine.run_batch_inference(model_name, run_id, batch) for batch in batches]
                batch_results = await asyncio.gather(*tasks)
                
                for br in batch_results:
                    all_predictions.extend(br)
            
            # 4. Storage and Exports
            await self.storage.save_predictions_batch(all_predictions)
            self.storage.export_to_csv(all_predictions, f"{run_name}_predictions.csv")
            self.storage.export_to_json(all_predictions, f"{run_name}_predictions.json")
            
            # 5. Mark Completed
            await self.storage.complete_benchmark_run(run_id, 'COMPLETED')
            logger.info(f"Benchmark run {run_name} completed successfully.")
            
        except Exception as e:
            logger.error(f"Benchmark run {run_name} failed: {e}")
            await self.storage.complete_benchmark_run(run_id, 'FAILED')
            raise
