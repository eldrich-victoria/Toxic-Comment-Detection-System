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

def predict_bert(tokenizer, model, device, texts, batch_size=32):

    model.eval()

    all_preds = []

    texts = list(texts)

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
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

        all_preds.extend(preds)

    return all_preds


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



# Traditional ML models struggle with contextual understanding,
# while BERT captures semantic meaning and significantly improves toxic detection.
# Although classical models achieved ~89% accuracy, their performance on the minority toxic class was poor (F1 ≈ 0.5). BERT significantly improved this to 0.90 by capturing contextual semantics, making it more suitable for real-world moderation systems.
# This project demonstrates that while traditional machine learning models like Logistic Regression and SVM provide fast and efficient baselines for toxicity detection, they struggle with contextual understanding and minority class prediction.
# In contrast, transformer-based models like BERT significantly outperform classical approaches by capturing semantic and contextual relationships in text, resulting in a substantial improvement in detecting toxic comments.
# However, this performance gain comes at the cost of increased computational complexity and inference time, highlighting the trade-off between efficiency and accuracy in real-world deployment scenarios.