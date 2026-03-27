import pandas as pd
import pickle
import time

from sklearn.metrics import classification_report, accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification
import torch


# -----------------------------
# LOAD TEST DATA
# -----------------------------

def load_test_data():
    print("Loading test data...")

    df = pd.read_csv("data/processed/cleaned_data.csv")

    df = df[["clean_text", "target"]].dropna()

    df = df.sample(n=20000, random_state=42)

    X = df["clean_text"]
    y = df["target"]

    return X, y


# -----------------------------
# LOAD ML MODELS
# -----------------------------

def load_ml_models():

    models = {
        "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
        "Linear SVM": pickle.load(open("models/linear_svm.pkl", "rb")),
    }

    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

    return models, vectorizer


# -----------------------------
# EVALUATE ML MODELS
# -----------------------------

def evaluate_ml(models, vectorizer, X, y):

    print("\n=== ML MODELS ===\n")

    X_vec = vectorizer.transform(X)

    results = {}

    for name, model in models.items():

        start = time.time()

        y_pred = model.predict(X_vec)

        end = time.time()

        acc = accuracy_score(y, y_pred)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print(classification_report(y, y_pred))

        results[name] = {
            "accuracy": acc,
            "time": end - start
        }

    return results


# -----------------------------
# LOAD BERT
# -----------------------------

def load_bert():
    print("\nLoading BERT model...")

    tokenizer = BertTokenizer.from_pretrained("models/bert")
    model = BertForSequenceClassification.from_pretrained("models/bert")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


# -----------------------------
# BERT PREDICTION
# -----------------------------

def predict_bert(tokenizer, model, device, texts):

    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    return preds


# -----------------------------
# EVALUATE BERT
# -----------------------------

def evaluate_bert(X, y):

    print("\n=== BERT MODEL ===\n")

    tokenizer, model, device = load_bert()

    start = time.time()

    y_pred = predict_bert(tokenizer, model, device, X)

    end = time.time()

    acc = accuracy_score(y, y_pred)

    print("BERT Accuracy:", acc)
    print(classification_report(y, y_pred))

    return {
        "accuracy": acc,
        "time": end - start
    }


# -----------------------------
# MAIN
# -----------------------------

def main():

    X, y = load_test_data()

    models, vectorizer = load_ml_models()

    ml_results = evaluate_ml(models, vectorizer, X, y)

    bert_results = evaluate_bert(X, y)

    print("\n=== FINAL SUMMARY ===\n")

    for model, res in ml_results.items():
        print(f"{model}: Accuracy={res['accuracy']:.4f}, Time={res['time']:.2f}s")

    print(f"BERT: Accuracy={bert_results['accuracy']:.4f}, Time={bert_results['time']:.2f}s")


if __name__ == "__main__":
    main()