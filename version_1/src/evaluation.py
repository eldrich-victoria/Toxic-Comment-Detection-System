import pickle
import os
import pandas as pd

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
        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        # Formatted classification report
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

        # Toxic class metrics
        # Class "1" represents the toxic class.
        f1_toxic = report_dict["1"]["f1-score"]
        recall_toxic = report_dict["1"]["recall"]
        precision_toxic = report_dict["1"]["precision"]

        # Store results
        results[name] = {
            "accuracy": acc,
            "f1_toxic": f1_toxic,
            "recall_toxic": recall_toxic,
            "precision_toxic": precision_toxic,
            "report": report,
            "confusion_matrix": cm
        }

    return results


# -----------------------------
# 5. SAVE RESULTS
# -----------------------------

def save_results(results):

    os.makedirs(
        os.path.join(
            get_version_path(),
            "outputs"
        ),
        exist_ok=True
    )

    rows = []

    for name, metrics in results.items():

        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_toxic": metrics["f1_toxic"],
            "recall_toxic": metrics["recall_toxic"],
            "precision_toxic": metrics["precision_toxic"]
        })

    results_df = pd.DataFrame(
        rows
    )

    output_file = os.path.join(
        get_version_path(),
        "outputs",
        "model_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\n📁 Results saved to:",
        output_file
    )

    print(
        "\n📊 FINAL COMPARISON:"
    )

    print(results_df)


# -----------------------------
# 6. MAIN
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

    # -----------------------------
    # SAVE RESULTS TO CSV
    # -----------------------------

    save_results(
        results
    )

    print(
        "\n🎯 Evaluation complete."
    )


# -----------------------------
# 7. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()