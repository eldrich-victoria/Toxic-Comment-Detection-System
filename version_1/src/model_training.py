import pickle
import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    print("Loading data...")
    X = pickle.load(open("models/X.pkl", "rb"))
    y = pickle.load(open("models/y.pkl", "rb"))
    return X, y


# -----------------------------
# TRAIN MODELS
# -----------------------------

def train_models(X_train, y_train):

    models = {}

    model_list = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )),

        ("Linear SVM", LinearSVC()),

        ("Random Forest", RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            n_jobs=-1
        ))
    ]

    for name, model in tqdm(model_list, desc="Training Models"):
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        print(f"✅ Finished {name}")
        models[name] = model

    return models


# -----------------------------
# SAVE MODELS
# -----------------------------

def save_models(models):
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        pickle.dump(model, open(f"models/{filename}", "wb"))

    print("\n✅ All models saved successfully.")


# -----------------------------
# MAIN
# -----------------------------

def main():
    X, y = load_data()

    # 🔥 MEMORY SAFE SAMPLING
    sample_size = 150000   # reduce to 100000 if needed
    print(f"\nUsing sample size: {sample_size}")

    X = X[:sample_size]
    y = y[:sample_size]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")
    models = train_models(X_train, y_train)

    save_models(models)

    print("\n🎯 Training complete.")


if __name__ == "__main__":
    main()



# Compared Logistic Regression, Linear SVM, and Random Forest models for toxicity classification. 
# Linear SVM performed best due to its effectiveness on high-dimensional sparse TF-IDF features.