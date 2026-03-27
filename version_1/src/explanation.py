import pickle
import re

from lime.lime_text import LimeTextExplainer


# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------

def load_objects():
    model = pickle.load(open("models/linear_svm.pkl", "rb"))
    vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))
    return model, vectorizer


# -----------------------------
# TEXT CLEANING (same as before)
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
# FEATURE-BASED EXPLANATION
# -----------------------------

BAD_WORDS = ["idiot", "stupid", "dumb", "hate", "kill", "moron"]

def feature_explanation(text):
    reasons = []

    words = text.split()

    for word in words:
        if word in BAD_WORDS:
            reasons.append(f"contains offensive word '{word}'")

    if len(reasons) == 0:
        return "No explicit toxic keywords detected."

    return "Flagged because it " + ", ".join(reasons)


# -----------------------------
# LIME EXPLANATION
# -----------------------------

def lime_explanation(text, model, vectorizer):

    class_names = ["Clean", "Toxic"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        X = vectorizer.transform(texts)
        # SVM does not have predict_proba → use decision function
        preds = model.decision_function(X)

        import numpy as np
        probs = 1 / (1 + np.exp(-preds))  # sigmoid
        return np.vstack([1 - probs, probs]).T

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6
    )

    return exp.as_list()


# -----------------------------
# MAIN PREDICT FUNCTION
# -----------------------------

def predict_and_explain(text):

    model, vectorizer = load_objects()

    clean = clean_text(text)
    X = vectorizer.transform([clean])

    pred = model.predict(X)[0]
    label = "Toxic" if pred == 1 else "Clean"

    # Feature explanation
    feat_exp = feature_explanation(clean)

    # LIME explanation
    lime_exp = lime_explanation(clean, model, vectorizer)

    return {
        "text": text,
        "prediction": label,
        "feature_explanation": feat_exp,
        "lime_explanation": lime_exp
    }


# -----------------------------
# TEST
# -----------------------------

if __name__ == "__main__":

    test_text = "You are such a stupid idiot"

    result = predict_and_explain(test_text)

    print("\n=== RESULT ===")
    print("Text:", result["text"])
    print("Prediction:", result["prediction"])
    print("\nFeature Explanation:")
    print(result["feature_explanation"])

    print("\nLIME Explanation:")
    for word, score in result["lime_explanation"]:
        print(f"{word}: {score:.3f}")