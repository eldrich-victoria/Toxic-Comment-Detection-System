import pickle
import re
import time
import logging
import numpy as np
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# INIT APP
# -----------------------------

app = FastAPI()

# -----------------------------
# REQUEST FORMAT
# -----------------------------

class InputText(BaseModel):
    text: str
    model_ids: list[str]
    normalize: bool = False
    enable_lime: bool = True

# -----------------------------
# DYNAMIC MODEL REGISTRY
# -----------------------------

MODEL_REGISTRY = {}
VECTORIZERS = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

pkl_models = {
    "v1_svm": ("version_1/models/linear_svm.pkl", "version_1/models/tfidf.pkl"),
    "v1_lr": ("version_1/models/logistic_regression.pkl", "version_1/models/tfidf.pkl"),
    "v1_rf": ("version_1/models/random_forest.pkl", "version_1/models/tfidf.pkl"),
    "v2_svm": ("version_2/models/linear_svm.pkl", "version_2/models/tfidf_vectorizer.pkl"),
    "v2_lr": ("version_2/models/logistic_regression.pkl", "version_2/models/tfidf_vectorizer.pkl"),
}

for model_id, (model_path, vec_path) in pkl_models.items():
    try:
        with open(model_path, "rb") as f:
            MODEL_REGISTRY[model_id] = pickle.load(f)
        with open(vec_path, "rb") as f:
            VECTORIZERS[model_id] = pickle.load(f)
        logger.info(f"Loaded {model_id} successfully.")
    except Exception as e:
        logger.warning(f"Failed to load {model_id}: {e}")

# Resilient BERT Loading
bert_path = "version_2/models/bert"
try:
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    bert_model.to(device)
    bert_model.eval()
    MODEL_REGISTRY["v2_bert"] = {"model": bert_model, "tokenizer": tokenizer}
    logger.info("Loaded BERT successfully.")
except Exception as e:
    logger.warning(f"Failed to load BERT from {bert_path}: {e}. Skipping.")

explainer = LimeTextExplainer(class_names=["Clean", "Toxic"])

# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text(text: str, normalize: bool = False):
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
    text = " ".join(text.split())
    return text

# -----------------------------
# FEATURE EXPLANATION
# -----------------------------

BAD_WORDS = ["idiot", "stupid", "dumb", "hate", "kill", "moron"]

def feature_explanation(text):
    reasons = []
    for word in text.split():
        if word in BAD_WORDS:
            reasons.append(f"contains offensive word '{word}'")

    if not reasons:
        return "No explicit toxic keywords detected."

    return "Flagged because it " + ", ".join(reasons)

# -----------------------------
# LIME EXPLANATION HELPER
# -----------------------------

def make_predict_proba_fn(m_id):
    def predict_proba(texts):
        if m_id == "v2_bert":
            bert_dict = MODEL_REGISTRY[m_id]
            tk = bert_dict["tokenizer"]
            mdl = bert_dict["model"]
            inputs = tk(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = mdl(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()
        else:
            model = MODEL_REGISTRY[m_id]
            vec = VECTORIZERS[m_id]
            X = vec.transform(texts)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                dist = model.decision_function(X)
                probs = 1 / (1 + np.exp(-dist))
                return np.vstack([1 - probs, probs]).T
    return predict_proba

# -----------------------------
# MAIN ENDPOINT
# -----------------------------

@app.post("/predict")
def predict(input: InputText):
    text = input.text
    clean = clean_text(text, input.normalize)
    feat_exp = feature_explanation(clean)

    results = {}
    
    for m_id in input.model_ids:
        if m_id not in MODEL_REGISTRY:
            continue
            
        start_t = time.time()
        
        # Prediction
        if m_id == "v2_bert":
            bert_dict = MODEL_REGISTRY[m_id]
            tk = bert_dict["tokenizer"]
            mdl = bert_dict["model"]
            inputs = tk(clean, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = mdl(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                prob = probs[1].item()
            pred_idx = 1 if prob >= 0.5 else 0
        else:
            model = MODEL_REGISTRY[m_id]
            vec = VECTORIZERS[m_id]
            X = vec.transform([clean])
            pred_idx = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0][1]
            else:
                dist = model.decision_function(X)[0]
                prob = 1 / (1 + np.exp(-dist))
                
        label = "Toxic" if pred_idx == 1 else "Clean"
        
        # LIME Explanation
        lime_exp = []
        if input.enable_lime and pred_idx == 1:
            try:
                predict_fn = make_predict_proba_fn(m_id)
                exp = explainer.explain_instance(clean, predict_fn, num_features=6)
                lime_exp = exp.as_list()
            except Exception as e:
                logger.error(f"LIME failed for {m_id}: {e}")
                
        latency = time.time() - start_t
        
        results[m_id] = {
            "prediction": label,
            "confidence": float(prob),
            "feature_explanation": feat_exp,
            "lime_explanation": lime_exp,
            "latency": f"{latency:.3f}s"
        }

    return results