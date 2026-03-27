import pickle
import os

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    print("Loading data...")
    X = pickle.load(open("models/X.pkl", "rb"))
    y = pickle.load(open("models/y.pkl", "rb"))
    return X, y


# -----------------------------
# LOAD MODELS
# -----------------------------

def load_models():
    models = {}

    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Linear SVM": "models/linear_svm.pkl",
        "Random Forest": "models/random_forest.pkl"
    }

    for name, path in model_files.items():
        models[name] = pickle.load(open(path, "rb"))

    return models


# -----------------------------
# EVALUATE MODELS
# -----------------------------

def evaluate(models, X_test, y_test):

    results = {}

    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

        results[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    return results


# -----------------------------
# MAIN
# -----------------------------

def main():
    X, y = load_data()

    # Same sampling as training
    sample_size = 150000
    X = X[:sample_size]
    y = y[:sample_size]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Loading models...")
    models = load_models()

    print("Evaluating models...")
    results = evaluate(models, X_test, y_test)

    print("\n🎯 Evaluation complete.")


if __name__ == "__main__":
    main()