import pickle
import re
import os
import numpy as np

from lime.lime_text import LimeTextExplainer


# -----------------------------
# 1. VERSION PATH
# -----------------------------

def get_version_path():
    """
    Returns the version_1 directory.

    Current file:
        Toxic-Comment-Detection-System/
        └── version_1/
            └── src/
                └── explanation.py

    Therefore:
        src -> version_1
    """

    return os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


# -----------------------------
# 2. LOAD MODEL + VECTORIZER
# -----------------------------

def load_objects():

    version_path = get_version_path()

    models_path = os.path.join(
        version_path,
        "models"
    )

    model_file = os.path.join(
        models_path,
        "linear_svm.pkl"
    )

    vectorizer_file = os.path.join(
        models_path,
        "tfidf.pkl"
    )

    print(
        "Loading model from:",
        model_file
    )

    print(
        "Loading vectorizer from:",
        vectorizer_file
    )

    with open(
        model_file,
        "rb"
    ) as file:

        model = pickle.load(file)

    with open(
        vectorizer_file,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    return model, vectorizer


# -----------------------------
# 3. TEXT CLEANING
# -----------------------------

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtags
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove emojis and special characters
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove extra spaces
    text = " ".join(
        text.split()
    )

    return text


# -----------------------------
# 4. FEATURE-BASED EXPLANATION
# -----------------------------

BAD_WORDS = [
    "idiot",
    "stupid",
    "dumb",
    "hate",
    "kill",
    "moron"
]


def feature_explanation(text):

    reasons = []

    words = text.split()

    for word in words:

        if word in BAD_WORDS:

            reasons.append(
                "contains offensive word '{}'".format(
                    word
                )
            )

    if len(reasons) == 0:

        return "No explicit toxic keywords detected."

    return (
        "Flagged because it "
        + ", ".join(reasons)
    )


# -----------------------------
# 5. LIME EXPLANATION
# -----------------------------

def lime_explanation(
    text,
    model,
    vectorizer
):

    class_names = [
        "Clean",
        "Toxic"
    ]

    explainer = LimeTextExplainer(
        class_names=class_names
    )

    def predict_proba(texts):

        X = vectorizer.transform(
            texts
        )

        # Linear SVM does not provide
        # predict_proba(), so use the
        # decision function.

        preds = model.decision_function(
            X
        )

        # Convert decision scores to
        # probability-like values using
        # sigmoid transformation.

        probs = 1 / (
            1 + np.exp(-preds)
        )

        return np.vstack(
            [
                1 - probs,
                probs
            ]
        ).T

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6
    )

    return exp.as_list()


# -----------------------------
# 6. MAIN PREDICT FUNCTION
# -----------------------------

def predict_and_explain(text):

    # Load model and vectorizer
    model, vectorizer = load_objects()

    # Clean input text
    clean = clean_text(text)

    # Convert text to TF-IDF features
    X = vectorizer.transform(
        [clean]
    )

    # Make prediction
    pred = model.predict(X)[0]

    label = (
        "Toxic"
        if pred == 1
        else "Clean"
    )

    # Feature-based explanation
    feat_exp = feature_explanation(
        clean
    )

    # LIME explanation
    lime_exp = lime_explanation(
        clean,
        model,
        vectorizer
    )

    return {
        "text": text,
        "prediction": label,
        "feature_explanation": feat_exp,
        "lime_explanation": lime_exp
    }


# -----------------------------
# 7. TEST
# -----------------------------

if __name__ == "__main__":

    test_text = (
        "You are such a stupid idiot"
    )

    result = predict_and_explain(
        test_text
    )

    print(
        "\n=== RESULT ==="
    )

    print(
        "Text:",
        result["text"]
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "\nFeature Explanation:"
    )

    print(
        result["feature_explanation"]
    )

    print(
        "\nLIME Explanation:"
    )

    for word, score in result[
        "lime_explanation"
    ]:

        print(
            "{}: {:.3f}".format(
                word,
                score
            )
        )