"""Inference engine for dynamic model loading and async execution."""
import asyncio
import time
from typing import List, Dict, Any

from app.core.logger import logger
from app.core.exceptions import ModelInitializationError

class InferenceEngine:
    """Engine for loading models dynamically and executing async batch inference."""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        
    def discover_and_load_models(self, model_configs: List[Dict[str, Any]]):
        """Automatically discover and load models based on config."""
        logger.info(f"Discovering and loading {len(model_configs)} models...")
        for config in model_configs:
            model_name = config.get("name", "unknown_model")
            model_type = config.get("type")
            model_version = config.get("version", "1.0")
            
            try:
                if model_type == "sklearn":
                    self._load_sklearn_model(model_name, config)
                elif model_type == "huggingface":
                    self._load_hf_model(model_name, config)
                else:
                    logger.warning(f"Unsupported model type '{model_type}' for {model_name}")
                    continue
                    
                self.loaded_models[model_name] = {
                    "type": model_type,
                    "version": model_version,
                    "config": config,
                }
                logger.info(f"Successfully loaded model: {model_name} (v{model_version})")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise ModelInitializationError(f"Could not load {model_name}") from e

    def _load_sklearn_model(self, name: str, config: Dict[str, Any]):
        """Placeholder for actual sklearn loading logic with vectorizer auto-matching."""
        logger.debug(f"Simulating loading sklearn model {name} with vectorizer from {config.get('path')}")

    def _load_hf_model(self, name: str, config: Dict[str, Any]):
        """Placeholder for actual huggingface loading logic."""
        logger.debug(f"Simulating loading HF model {name} from {config.get('path')}")

    def normalize_text(self, text: str) -> str:
        """Text normalization before inference."""
        return text.lower().strip()

    async def run_batch_inference(
        self, 
        model_name: str, 
        run_id: int, 
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run asynchronous batch inference tracking latency and confidence."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} is not loaded.")
            
        model_info = self.loaded_models[model_name]
        results = []
        
        start_time = time.time()
        # Simulate async inference processing depending on batch size
        await asyncio.sleep(0.01 * len(batch)) 
        
        for item in batch:
            text = item.get("text", "")
            sample_id = item.get("id", "unknown")
            ground_truth = item.get("ground_truth")
            
            normalized = self.normalize_text(text)
            
            # Simulated inference logic
            prediction = 0.85 if "toxic" in normalized else 0.15
            confidence = 0.90
            latency_ms = (time.time() - start_time) * 1000 / len(batch)
            
            results.append({
                "run_id": run_id,
                "model_name": model_name,
                "model_version": model_info["version"],
                "sample_id": sample_id,
                "text_input": text,
                "normalized_text": normalized,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "xai_placeholder": "{}" 
            })
            
        return results
