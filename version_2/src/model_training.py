import pickle
import os

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    print("Loading training data...")

    X_train = pickle.load(open("models/X_train.pkl", "rb"))
    y_train = pickle.load(open("models/y_train.pkl", "rb"))

    return X_train, y_train


# -----------------------------
# TRAIN MODELS WITH CHECKPOINT
# -----------------------------

def train_models(X_train, y_train):

    checkpoint_path = "models/checkpoint.pkl"

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        models = pickle.load(open(checkpoint_path, "rb"))
    else:
        models = {}

    model_list = [
        ("logistic_regression", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        )),

        ("linear_svm", LinearSVC(
            class_weight="balanced"
        ))
    ]

    for name, model in tqdm(model_list, desc="Training Models"):

        if name in models:
            print(f"Skipping {name}, already trained.")
            continue

        print(f"\n🚀 Training {name}...")

        model.fit(X_train, y_train)

        print(f"✅ Finished {name}")

        models[name] = model

        # Save checkpoint after each model
        pickle.dump(models, open(checkpoint_path, "wb"))

    return models


# -----------------------------
# SAVE FINAL MODELS
# -----------------------------

def save_models(models):

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        path = f"models/{name}.pkl"
        pickle.dump(model, open(path, "wb"))

    print("\n✅ All models saved successfully.")


# -----------------------------
# MAIN
# -----------------------------

def main():

    X_train, y_train = load_data()

    print("Training models...")
    models = train_models(X_train, y_train)

    save_models(models)

    print("\n🎯 Training complete.")


if __name__ == "__main__":
    main()