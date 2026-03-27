import pickle
import os
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    print("Loading test data...")

    X_test = pickle.load(open("models/X_test.pkl", "rb"))
    y_test = pickle.load(open("models/y_test.pkl", "rb"))

    return X_test, y_test


# -----------------------------
# LOAD MODELS
# -----------------------------

def load_models():

    models = {}

    model_paths = {
        "logistic_regression": "models/logistic_regression.pkl",
        "linear_svm": "models/linear_svm.pkl"
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = pickle.load(open(path, "rb"))
        else:
            print(f"⚠️ Model not found: {name}")

    return models


# -----------------------------
# EVALUATE MODELS
# -----------------------------

def evaluate(models, X_test, y_test):

    results = []

    for name, model in models.items():

        print(f"\n🔍 Evaluating {name}...")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        f1_toxic = report["1"]["f1-score"]
        recall_toxic = report["1"]["recall"]
        precision_toxic = report["1"]["precision"]

        print(f"\n{name.upper()} RESULTS")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (Toxic): {f1_toxic:.4f}")
        print(f"Recall (Toxic): {recall_toxic:.4f}")
        print(f"Precision (Toxic): {precision_toxic:.4f}")
        print("Confusion Matrix:")
        print(cm)

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_toxic": f1_toxic,
            "recall_toxic": recall_toxic,
            "precision_toxic": precision_toxic
        })

    return pd.DataFrame(results)


# -----------------------------
# SAVE RESULTS
# -----------------------------

def save_results(df):

    os.makedirs("outputs", exist_ok=True)

    df.to_csv("outputs/model_comparison.csv", index=False)

    print("\n📁 Results saved to outputs/model_comparison.csv")


# -----------------------------
# MAIN
# -----------------------------

def main():

    X_test, y_test = load_data()

    models = load_models()

    results_df = evaluate(models, X_test, y_test)

    print("\n📊 FINAL COMPARISON:")
    print(results_df)

    save_results(results_df)


if __name__ == "__main__":
    main()