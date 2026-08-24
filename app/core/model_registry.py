import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized registry for loading and managing ML models safely."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.artifacts_dir = self.root_dir / "artifacts"
        
        self.model_configs = {
            "v1_svm": (self.artifacts_dir / "v1_svm/linear_svm.pkl", self.artifacts_dir / "v1_svm/tfidf.pkl"),
            "v1_lr": (self.artifacts_dir / "v1_lr/logistic_regression.pkl", self.artifacts_dir / "v1_lr/tfidf.pkl"),
            "v1_rf": (self.artifacts_dir / "v1_rf/random_forest.pkl", self.artifacts_dir / "v1_rf/tfidf.pkl"),
            "v2_svm": (self.artifacts_dir / "v2_svm/linear_svm.pkl", self.artifacts_dir / "v2_svm/tfidf_vectorizer.pkl"),
            "v2_lr": (self.artifacts_dir / "v2_lr/logistic_regression.pkl", self.artifacts_dir / "v2_lr/tfidf_vectorizer.pkl"),
        }
        
    def get_available_models(self) -> List[str]:
        """Return list of models that could be successfully loaded."""
        return list(self.models.keys())
        
    def get_model_statuses(self) -> Dict[str, bool]:
        """Return dictionary mapping all known models to their availability status."""
        statuses = {}
        for model_id in self.model_configs:
            statuses[model_id] = model_id in self.models
        statuses["v2_bert"] = "v2_bert" in self.models
        return statuses
        
    def load_all_models(self):
        """Attempts to load all configured models and vectorizers."""
        for model_id, (model_path, vec_path) in self.model_configs.items():
            try:
                
                if model_path.exists() and vec_path.exists():
                    with open(model_path, "rb") as f:
                        self.models[model_id] = pickle.load(f)
                    with open(vec_path, "rb") as f:
                        self.vectorizers[model_id] = pickle.load(f)
                    logger.info(f"Loaded {model_id} successfully.")
                else:
                    logger.warning(f"Files missing for {model_id}. Skipping.")
            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")
                
        # Attempt to load BERT
        bert_path = self.artifacts_dir / "v2_bert"
        if bert_path.exists() and (
            (bert_path / "pytorch_model.bin").exists() or 
            (bert_path / "model.safetensors").exists()
        ):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
                bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_path))
                bert_model.to(self.device)
                bert_model.eval()
                self.models["v2_bert"] = {"model": bert_model, "tokenizer": tokenizer}
                logger.info("Loaded BERT successfully.")
            except Exception as e:
                logger.warning(f"Failed to load BERT from {bert_path}: {e}. Skipping.")
        else:
            logger.warning("BERT model weights not found in repository. Skipping BERT loading.")

    def _predict_bert(self, text: str) -> Dict[str, Any]:
        bert_dict = self.models["v2_bert"]
        tokenizer = bert_dict["tokenizer"]
        model = bert_dict["model"]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            prob = probs[1].item()
            
        pred_idx = 1 if prob >= 0.5 else 0
        return {"prediction": pred_idx, "confidence": float(prob)}

    def predict(self, text: str, model_id: str) -> Dict[str, Any]:
        """Run inference for a single text using a specific model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not loaded.")
            
        if model_id == "v2_bert":
            return self._predict_bert(text)
            
        # Classical models
        model = self.models[model_id]
        vec = self.vectorizers[model_id]
        
        X = vec.transform([text])
        pred_idx = int(model.predict(X)[0])
        
        prob = 0.0
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
        elif hasattr(model, "decision_function"):
            dist = model.decision_function(X)[0]
            prob = float(1 / (1 + np.exp(-dist)))
            
        return {"prediction": pred_idx, "confidence": prob}
        
    def predict_proba_for_lime(self, texts: List[str], model_id: str) -> np.ndarray:
        """Prediction function required for LIME explanation."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not loaded.")
            
        if model_id == "v2_bert":
            bert_dict = self.models["v2_bert"]
            tokenizer = bert_dict["tokenizer"]
            model = bert_dict["model"]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()
                
        # Classical models
        model = self.models[model_id]
        vec = self.vectorizers[model_id]
        X = vec.transform(texts)
        
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        else:
            dist = model.decision_function(X)
            probs = 1 / (1 + np.exp(-dist))
            return np.vstack([1 - probs, probs]).T

# Global singleton to avoid reloading models on every request
registry = None

def get_registry(root_dir: Path) -> ModelRegistry:
    global registry
    if registry is None:
        registry = ModelRegistry(root_dir)
        registry.load_all_models()
    return registry
