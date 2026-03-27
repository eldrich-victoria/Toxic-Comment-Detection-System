import pickle
import re
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from lime.lime_text import LimeTextExplainer


# -----------------------------
# INIT APP
# -----------------------------

app = FastAPI()


# -----------------------------
# REQUEST FORMAT
# -----------------------------

class InputText(BaseModel):
    text: str


# -----------------------------
# LOAD MODEL + VECTORIZER (ONCE)
# -----------------------------

model = pickle.load(open("models/linear_svm.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

explainer = LimeTextExplainer(class_names=["Clean", "Toxic"])


# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text(text):
    text = text.lower()
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
# LIME EXPLANATION
# -----------------------------

def predict_proba(texts):
    X = vectorizer.transform(texts)
    preds = model.decision_function(X)
    probs = 1 / (1 + np.exp(-preds))
    return np.vstack([1 - probs, probs]).T


def lime_explanation(text):
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6
    )
    return exp.as_list()


# -----------------------------
# MAIN ENDPOINT
# -----------------------------

@app.post("/predict")
def predict(input: InputText):

    text = input.text
    clean = clean_text(text)

    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]

    label = "Toxic" if pred == 1 else "Clean"

    # Feature explanation
    feat_exp = feature_explanation(clean)

    # LIME only if toxic (optimization)
    lime_exp = lime_explanation(clean) if pred == 1 else []

    return {
        "text": text,
        "prediction": label,
        "feature_explanation": feat_exp,
        "lime_explanation": lime_exp
    }