import uuid
import re
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
import pandas as pd
import yaml
import time
import json
from lime.lime_text import LimeTextExplainer

from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.inference_engine import InferenceEngine
from database.storage_manager import StorageManager
from app.core.model_registry import get_registry

router = APIRouter(tags=["API"])
explainer = LimeTextExplainer(class_names=["Clean", "Toxic"])

# Models
class InputText(BaseModel):
    text: str
    model_ids: List[str]
    normalize: bool = False
    enable_lime: bool = True

def clean_text(text: str, normalize: bool = False) -> str:
    text = text.lower()
    if normalize:
        # Handle leetspeak/homoglyphs for adversarial detection
        replacements = {
            '@': 'a', '0': 'o', '1': 'i', '!': 'i', '3': 'e',
            '$': 's', '5': 's', '7': 't'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
            
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return " ".join(text.split())

def feature_explanation(text: str) -> str:
    BAD_WORDS = ["idiot", "stupid", "dumb", "hate", "kill", "moron"]
    reasons = [f"contains offensive word '{word}'" for word in text.split() if word in BAD_WORDS]
    if not reasons:
        return "No explicit toxic keywords detected."
    return "Flagged because it " + ", ".join(reasons)

def normalize_target_label(val: Any) -> int:
    if pd.isna(val):
        return None
    
    if isinstance(val, (int, float)):
        if val == 1:
            return 1
        elif val == 0:
            return 0
            
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in ["1", "toxic", "true", "yes"]:
            return 1
        elif val_lower in ["0", "not toxic", "false", "no"]:
            return 0
            
    raise HTTPException(
        status_code=400,
        detail=f"Invalid target value: '{val}'. Accepted values are 1/0, 'Toxic'/'Not Toxic', 'True'/'False', 'Yes'/'No'."
    )

@router.get("/models")
async def get_models():
    """Return available models in the registry."""
    registry = get_registry()
    return {
        "models": registry.get_available_models(),
        "statuses": registry.get_model_statuses()
    }

@router.get("/production-model")
async def get_production_model():
    """Return the production model manifest."""
    manifest_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "production_model.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=500, detail="Production model manifest not found.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest

@router.post("/predict")
async def predict_text(input: InputText):
    """Run real-time prediction and optional LIME explanation."""
    registry = get_registry()
    clean = clean_text(input.text, input.normalize)
    feat_exp = feature_explanation(clean)
    
    results = {}
    available_models = registry.get_available_models()
    
    for m_id in input.model_ids:
        if m_id not in available_models:
            results[m_id] = {"error": f"Required local model artifact for {m_id} is missing."}
            continue
            
        start_t = time.time()
        
        try:
            # Predict
            pred_res = registry.predict(clean, m_id)
            pred_idx = pred_res["prediction"]
            prob = pred_res["confidence"]
            label = "Toxic" if pred_idx == 1 else "Clean"
            
            # LIME Explanation
            lime_exp = []
            if input.enable_lime and pred_idx == 1:
                try:
                    def predict_fn(texts):
                        return registry.predict_proba_for_lime(texts, m_id)
                    exp = explainer.explain_instance(clean, predict_fn, num_features=6)
                    lime_exp = exp.as_list()
                except Exception as e:
                    print(f"LIME failed for {m_id}: {e}")
                    
            latency = time.time() - start_t
            
            results[m_id] = {
                "prediction": label,
                "confidence": prob,
                "feature_explanation": feat_exp,
                "lime_explanation": lime_exp,
                "latency": f"{latency:.3f}s"
            }
        except Exception as e:
            print(f"Prediction failed for {m_id}: {e}")
            results[m_id] = {"error": str(e)}

    return results

@router.post("/benchmark/run")
async def run_benchmark(file: UploadFile = File(...)):
    try:
        print("\n========== BENCHMARK STARTED ==========\n")

        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        project_root = Path(__file__).resolve().parent.parent.parent
        temp_dir = project_root / "temp_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Secure file naming to prevent path traversal
        safe_filename = f"{uuid.uuid4()}_{re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)}"
        temp_file_path = temp_dir / safe_filename

        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        df = pd.read_csv(temp_file_path)
        
        possible_columns = ["comment", "comments", "text", "sentence", "content", "clean_text"]
        target_columns = ["target", "toxic", "is_toxic", "label", "ground_truth"]
        
        comment_column = next((col for col in possible_columns if col in df.columns), None)
        target_column = next((col for col in target_columns if col in df.columns), None)

        if comment_column is None:
            raise HTTPException(status_code=400, detail=f"No valid comment column found. Expected one of: {possible_columns}")

        dataset = []
        for idx, row in df.iterrows():
            gt = None
            if target_column:
                gt = normalize_target_label(row[target_column])
            
            dataset.append({
                "id": str(idx),
                "text": str(row[comment_column]),
                "ground_truth": gt
            })

        config_path = project_root / "configs" / "model_config.yaml"
        if not config_path.exists():
            raise HTTPException(status_code=500, detail="model_config.yaml not found.")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        model_configs = config_data.get("models", [])
        benchmark_config = config_data.get("benchmark", {})

        if not model_configs:
            raise HTTPException(status_code=500, detail="No models found in model_config.yaml")

        inference_engine = InferenceEngine(project_root)
        storage_manager = StorageManager()
        await storage_manager.init_db()

        benchmark_runner = BenchmarkRunner(
            inference_engine=inference_engine,
            storage_manager=storage_manager,
            config={"benchmark": benchmark_config}
        )

        await benchmark_runner.run_benchmark(dataset=dataset, model_configs=model_configs)

        print("\n========== BENCHMARK FINISHED ==========\n")

        # Cleanup temp file
        temp_file_path.unlink(missing_ok=True)

        return {
            "status": "success",
            "message": "Benchmark completed successfully.",
            "dataset_size": len(dataset),
            "models_used": [m["name"] for m in model_configs]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print("\n========== ERROR ==========")
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))