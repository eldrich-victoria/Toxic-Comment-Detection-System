import pickle
import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


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
                └── model_training.py

    Therefore:
        src -> version_1
    """

    return os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


# -----------------------------
# 2. LOAD DATA
# -----------------------------

def load_data():

    print("Loading data...")

    version_path = get_version_path()

    models_path = os.path.join(
        version_path,
        "models"
    )

    X_file = os.path.join(
        models_path,
        "X.pkl"
    )

    y_file = os.path.join(
        models_path,
        "y.pkl"
    )

    print("Loading features from:", X_file)
    print("Loading targets from:", y_file)

    with open(
        X_file,
        "rb"
    ) as file:

        X = pickle.load(file)

    with open(
        y_file,
        "rb"
    ) as file:

        y = pickle.load(file)

    return X, y


# -----------------------------
# 3. TRAIN MODELS
# -----------------------------

def train_models(
    X_train,
    y_train
):

    models = {}

    model_list = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1
            )
        ),

        (
            "Linear SVM",
            LinearSVC()
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=20,
                n_jobs=-1
            )
        )
    ]

    for name, model in tqdm(
        model_list,
        desc="Training Models"
    ):

        print(
            "\n🚀 Training",
            name + "..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "✅ Finished",
            name
        )

        models[name] = model

    return models


# -----------------------------
# 4. SAVE MODELS
# -----------------------------

def save_models(models):

    version_path = get_version_path()

    models_path = os.path.join(
        version_path,
        "models"
    )

    # Create models directory if it does not exist
    os.makedirs(
        models_path,
        exist_ok=True
    )

    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        output_file = os.path.join(
            models_path,
            filename
        )

        with open(
            output_file,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        print(
            "Saved:",
            output_file
        )

    print(
        "\n✅ All models saved successfully."
    )


# -----------------------------
# 5. MAIN
# -----------------------------

def main():

    # Load TF-IDF features and targets
    X, y = load_data()

    # -----------------------------
    # MEMORY SAFE SAMPLING
    # -----------------------------

    sample_size = 150000

    print(
        "\nUsing sample size:",
        sample_size
    )

    X = X[:sample_size]
    y = y[:sample_size]

    print(
        "Sampled data shape:",
        X.shape
    )

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------

    print(
        "Splitting data..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------

    print(
        "Training models..."
    )

    models = train_models(
        X_train,
        y_train
    )

    # -----------------------------
    # SAVE MODELS
    # -----------------------------

    save_models(
        models
    )

    print(
        "\n🎯 Training complete."
    )


# -----------------------------
# 6. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()