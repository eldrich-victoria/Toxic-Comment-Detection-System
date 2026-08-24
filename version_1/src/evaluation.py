import pickle
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import train_test_split


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
                └── evaluation.py

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

    print(
        "Loading features from:",
        X_file
    )

    print(
        "Loading targets from:",
        y_file
    )

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
# 3. LOAD MODELS
# -----------------------------

def load_models():

    models = {}

    version_path = get_version_path()

    models_path = os.path.join(
        version_path,
        "models"
    )

    model_files = {
        "Logistic Regression": os.path.join(
            models_path,
            "logistic_regression.pkl"
        ),

        "Linear SVM": os.path.join(
            models_path,
            "linear_svm.pkl"
        ),

        "Random Forest": os.path.join(
            models_path,
            "random_forest.pkl"
        )
    }

    for name, path in model_files.items():

        print(
            "Loading:",
            path
        )

        with open(
            path,
            "rb"
        ) as file:

            models[name] = pickle.load(
                file
            )

    return models


# -----------------------------
# 4. EVALUATE MODELS
# -----------------------------

def evaluate(
    models,
    X_test,
    y_test
):

    results = {}

    for name, model in models.items():

        print(
            "\n🔍 Evaluating",
            name + "..."
        )

        # Generate predictions
        y_pred = model.predict(
            X_test
        )

        # Accuracy
        acc = accuracy_score(
            y_test,
            y_pred
        )

        # Classification report
        report = classification_report(
            y_test,
            y_pred
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print(
            "\n{} Accuracy: {:.4f}".format(
                name,
                acc
            )
        )

        print(
            "\nClassification Report:"
        )

        print(report)

        print(
            "Confusion Matrix:"
        )

        print(cm)

        results[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    return results


# -----------------------------
# 5. MAIN
# -----------------------------

def main():

    # -----------------------------
    # LOAD DATA
    # -----------------------------

    X, y = load_data()

    # -----------------------------
    # SAME SAMPLING AS TRAINING
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
    # LOAD MODELS
    # -----------------------------

    print(
        "Loading models..."
    )

    models = load_models()

    # -----------------------------
    # EVALUATE MODELS
    # -----------------------------

    print(
        "Evaluating models..."
    )

    results = evaluate(
        models,
        X_test,
        y_test
    )

    print(
        "\n🎯 Evaluation complete."
    )


# -----------------------------
# 6. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()