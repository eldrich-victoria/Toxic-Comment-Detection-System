import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification
)


# ============================================================
# PROJECT PATHS
# ============================================================

VERSION_2_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = VERSION_2_DIR.parent

MODELS_DIR = VERSION_2_DIR / "models"

OUTPUTS_DIR = VERSION_2_DIR / "outputs"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

LOGISTIC_PATH = (
    MODELS_DIR
    / "logistic_regression.pkl"
)

SVM_PATH = (
    MODELS_DIR
    / "linear_svm.pkl"
)

VECTORIZER_PATH = (
    MODELS_DIR
    / "tfidf_vectorizer.pkl"
)

X_TEST_PATH = (
    MODELS_DIR
    / "X_test.pkl"
)

Y_TEST_PATH = (
    MODELS_DIR
    / "y_test.pkl"
)

BERT_FINAL_DIR = (
    MODELS_DIR
    / "bert"
    / "final"
)


# ============================================================
# FEATURE ENGINEERING SETTINGS
#
# THESE MUST MATCH feature_engineering.py
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# BERT SETTINGS
# ============================================================

BERT_MAX_LENGTH = 128

BERT_BATCH_SIZE = 32


# ============================================================
# UTILITY
# ============================================================

def verify_file(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "Required file not found:\n"
            + str(path)
        )

    if path.stat().st_size == 0:

        raise ValueError(
            "Required file is empty:\n"
            + str(path)
        )


# ============================================================
# LOAD AND RECONSTRUCT EXACT TEST TEXT
# ============================================================

def load_exact_test_text():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECONSTRUCTING EXACT TEST TEXT"
    )

    print(
        "=" * 70
    )

    print(
        "\nSource dataset:"
    )

    print(
        DATA_PATH
    )

    verify_file(
        DATA_PATH
    )

    print(
        "\nLoading source dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Original dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # MATCH feature_engineering.py EXACTLY
    # --------------------------------------------------------

    required_columns = [
        "clean_text",
        "target"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if len(missing_columns) > 0:

        raise KeyError(
            "Required columns are missing:\n"
            + str(missing_columns)
        )

    # Keep only the required columns
    df = df[
        [
            "clean_text",
            "target"
        ]
    ].copy()

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    df["clean_text"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # CLEAN TARGET
    # --------------------------------------------------------

    df["target"] = (
        pd.to_numeric(
            df["target"],
            errors="coerce"
        )
    )

    # Remove invalid target values
    df = df.dropna(
        subset=["target"]
    )

    df["target"] = (
        df["target"]
        .astype(int)
    )

    # Remove empty comments
    df = df[
        df["clean_text"]
        .str.strip()
        .ne("")
    ]

    # Keep binary target classes
    df = df[
        df["target"].isin(
            [0, 1]
        )
    ]

    # Remove duplicate comments
    df = df.drop_duplicates(
        subset=[
            "clean_text",
            "target"
        ]
    )

    print(
        "\nDataset after applying the same "
        "feature-engineering preprocessing:"
    )

    print(
        df.shape
    )

    # --------------------------------------------------------
    # SAME STRATIFIED SPLIT
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    X = df["clean_text"]

    y = df["target"]

    print(
        "\nReconstructing stratified 80/20 split..."
    )

    (
        X_train_text,
        X_test_text,
        y_train_text,
        y_test_text
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "Reconstructed test samples:",
        len(X_test_text)
    )

    print(
        "Reconstructed test labels:",
        len(y_test_text)
    )

    return (
        X_test_text.reset_index(drop=True),
        y_test_text.reset_index(drop=True)
    )


# ============================================================
# LOAD SAVED TEST FEATURES / LABELS
# ============================================================

def load_saved_test_artifacts():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING SAVED TEST ARTIFACTS"
    )

    print(
        "=" * 70
    )

    verify_file(
        X_TEST_PATH
    )

    verify_file(
        Y_TEST_PATH
    )

    print(
        "\nLoading:",
        X_TEST_PATH
    )

    with open(
        X_TEST_PATH,
        "rb"
    ) as file:

        X_test = pickle.load(
            file
        )

    print(
        "Loading:",
        Y_TEST_PATH
    )

    with open(
        Y_TEST_PATH,
        "rb"
    ) as file:

        y_test = pickle.load(
            file
        )

    y_test = np.asarray(
        y_test
    )

    print(
        "\nX_test shape:",
        X_test.shape
    )

    print(
        "y_test shape:",
        y_test.shape
    )

    if X_test.shape[0] != len(y_test):

        raise ValueError(
            "X_test and y_test contain different "
            "numbers of samples."
        )

    return (
        X_test,
        y_test
    )


# ============================================================
# VERIFY EXACT TEST SET ALIGNMENT
# ============================================================

def verify_test_alignment(
    reconstructed_text,
    reconstructed_labels,
    saved_labels
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERIFYING TEST SET ALIGNMENT"
    )

    print(
        "=" * 70
    )

    print(
        "\nReconstructed test text samples:",
        len(reconstructed_text)
    )

    print(
        "Reconstructed test labels:",
        len(reconstructed_labels)
    )

    print(
        "Saved y_test samples:",
        len(saved_labels)
    )

    # --------------------------------------------------------
    # SAMPLE COUNT
    # --------------------------------------------------------

    if len(reconstructed_text) != len(saved_labels):

        raise ValueError(
            "\nTest-set size mismatch.\n"
            "Reconstructed text samples: "
            + str(len(reconstructed_text))
            + "\nSaved y_test samples: "
            + str(len(saved_labels))
        )

    # --------------------------------------------------------
    # LABEL ALIGNMENT
    # --------------------------------------------------------

    reconstructed_array = (
        reconstructed_labels
        .to_numpy()
    )

    if not np.array_equal(
        reconstructed_array,
        saved_labels
    ):

        mismatches = np.where(
            reconstructed_array
            != saved_labels
        )[0]

        raise ValueError(
            "\nTEST SET ALIGNMENT FAILED.\n"
            "The reconstructed test labels do not match "
            "the saved y_test labels.\n"
            "Number of mismatches: "
            + str(len(mismatches))
        )

    print(
        "\n✅ Test labels match exactly."
    )

    print(
        "✅ Test sample count matches exactly."
    )

    print(
        "\nThe same test split will be used for:"
    )

    print(
        "  Logistic Regression"
    )

    print(
        "  Linear SVM"
    )

    print(
        "  BERT"
    )

    return (
        reconstructed_text.tolist(),
        saved_labels
    )


# ============================================================
# LOAD CLASSICAL MODELS
# ============================================================

def load_ml_models():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING CLASSICAL ML MODELS"
    )

    print(
        "=" * 70
    )

    required_files = {
        "Logistic Regression": LOGISTIC_PATH,
        "Linear SVM": SVM_PATH,
        "TF-IDF Vectorizer": VECTORIZER_PATH
    }

    for name, path in required_files.items():

        verify_file(
            path
        )

        print(
            "Verified:",
            name
        )

    print(
        "\nLoading Logistic Regression..."
    )

    with open(
        LOGISTIC_PATH,
        "rb"
    ) as file:

        logistic_model = pickle.load(
            file
        )

    print(
        "Loading Linear SVM..."
    )

    with open(
        SVM_PATH,
        "rb"
    ) as file:

        svm_model = pickle.load(
            file
        )

    print(
        "Loading TF-IDF vectorizer..."
    )

    with open(
        VECTORIZER_PATH,
        "rb"
    ) as file:

        vectorizer = pickle.load(
            file
        )

    models = {
        "Logistic Regression": logistic_model,
        "Linear SVM": svm_model
    }

    print(
        "\nClassical ML models loaded successfully."
    )

    return (
        models,
        vectorizer
    )


# ============================================================
# VERIFY ML MODEL COMPATIBILITY
# ============================================================

def verify_ml_compatibility(
    models,
    vectorizer,
    X_test
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERIFYING ML MODEL COMPATIBILITY"
    )

    print(
        "=" * 70
    )

    expected_features = X_test.shape[1]

    print(
        "\nX_test feature count:",
        expected_features
    )

    # --------------------------------------------------------
    # VECTORISER
    # --------------------------------------------------------

    if not hasattr(
        vectorizer,
        "vocabulary_"
    ):

        raise RuntimeError(
            "TF-IDF vectorizer does not contain a vocabulary."
        )

    vectorizer_features = len(
        vectorizer.vocabulary_
    )

    print(
        "\nTF-IDF vectorizer features:",
        vectorizer_features
    )

    if vectorizer_features != expected_features:

        raise ValueError(
            "TF-IDF vectorizer and X_test are incompatible.\n"
            "Vectorizer features: "
            + str(vectorizer_features)
            + "\nX_test features: "
            + str(expected_features)
        )

    print(
        "TF-IDF compatibility: PASS"
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        if not hasattr(
            model,
            "n_features_in_"
        ):

            raise RuntimeError(
                name
                + " does not expose n_features_in_."
            )

        model_features = (
            model.n_features_in_
        )

        print(
            "\n"
            + name
        )

        print(
            "Model features:",
            model_features
        )

        print(
            "Test features:",
            expected_features
        )

        if model_features != expected_features:

            raise ValueError(
                name
                + " expects "
                + str(model_features)
                + " features, but X_test has "
                + str(expected_features)
            )

        print(
            "Compatibility: PASS"
        )


# ============================================================
# EVALUATE CLASSICAL MODEL
# ============================================================

def evaluate_classical_model(
    name,
    model,
    X_test,
    y_test
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Evaluating:",
        name
    )

    print(
        "-" * 70
    )

    start_time = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    end_time = time.perf_counter()

    inference_time = (
        end_time
        - start_time
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    result = {
        "model": name,
        "accuracy": report["accuracy"],
        "precision_toxic": report["1"]["precision"],
        "recall_toxic": report["1"]["recall"],
        "f1_toxic": report["1"]["f1-score"],
        "inference_time_seconds": inference_time
    }

    detailed = []

    for class_name in [
        "0",
        "1"
    ]:

        if class_name in report:

            detailed.append(
                {
                    "model": name,
                    "class": class_name,
                    "precision": report[class_name][
                        "precision"
                    ],
                    "recall": report[class_name][
                        "recall"
                    ],
                    "f1": report[class_name][
                        "f1-score"
                    ],
                    "support": report[class_name][
                        "support"
                    ]
                }
            )

    print(
        "\nAccuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Toxic Precision:",
        round(
            result["precision_toxic"],
            4
        )
    )

    print(
        "Toxic Recall:",
        round(
            result["recall_toxic"],
            4
        )
    )

    print(
        "Toxic F1:",
        round(
            result["f1_toxic"],
            4
        )
    )

    print(
        "Inference Time:",
        round(
            inference_time,
            4
        ),
        "seconds"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion
    )

    return (
        result,
        detailed,
        confusion
    )


# ============================================================
# LOAD BERT
# ============================================================

def load_bert():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING BERT"
    )

    print(
        "=" * 70
    )

    if not BERT_FINAL_DIR.exists():

        raise FileNotFoundError(
            "BERT final directory not found:\n"
            + str(BERT_FINAL_DIR)
        )

    model_path = (
        BERT_FINAL_DIR
        / "model.safetensors"
    )

    config_path = (
        BERT_FINAL_DIR
        / "config.json"
    )

    verify_file(
        model_path
    )

    verify_file(
        config_path
    )

    print(
        "\nBERT directory:"
    )

    print(
        BERT_FINAL_DIR
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT_FINAL_DIR)
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        "\nLoading BERT model..."
    )

    model = BertForSequenceClassification.from_pretrained(
        str(BERT_FINAL_DIR)
    )

    # --------------------------------------------------------
    # GPU REQUIRED
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available.\n"
            "BERT final evaluation requires the GPU."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nDevice:",
        device
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    model.to(
        device
    )

    model.eval()

    print(
        "\nBERT model loaded successfully."
    )

    return (
        tokenizer,
        model,
        device
    )


# ============================================================
# BERT PREDICTION
# ============================================================

def predict_bert(
    tokenizer,
    model,
    device,
    texts
):

    predictions = []

    total = len(
        texts
    )

    print(
        "\nTotal BERT samples:",
        total
    )

    print(
        "BERT batch size:",
        BERT_BATCH_SIZE
    )

    with torch.no_grad():

        for start in range(
            0,
            total,
            BERT_BATCH_SIZE
        ):

            end = min(
                start
                + BERT_BATCH_SIZE,
                total
            )

            batch_texts = texts[
                start:end
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(
                    device
                )
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            batch_predictions = (
                torch.argmax(
                    outputs.logits,
                    dim=1
                )
                .cpu()
                .numpy()
                .tolist()
            )

            predictions.extend(
                batch_predictions
            )

            processed = end

            if (
                processed % (
                    BERT_BATCH_SIZE * 10
                ) == 0
                or processed == total
            ):

                print(
                    "Processed:",
                    processed,
                    "/",
                    total
                )

    return np.asarray(
        predictions
    )


# ============================================================
# EVALUATE BERT
# ============================================================

def evaluate_bert(
    tokenizer,
    model,
    device,
    texts,
    y_test
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATING BERT"
    )

    print(
        "=" * 70
    )

    if len(texts) != len(y_test):

        raise ValueError(
            "BERT text count and label count do not match."
        )

    start_time = time.perf_counter()

    y_pred = predict_bert(
        tokenizer,
        model,
        device,
        texts
    )

    # Make sure all GPU operations have completed
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = (
        end_time
        - start_time
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    result = {
        "model": "BERT",
        "accuracy": report["accuracy"],
        "precision_toxic": report["1"]["precision"],
        "recall_toxic": report["1"]["recall"],
        "f1_toxic": report["1"]["f1-score"],
        "inference_time_seconds": inference_time
    }

    detailed = []

    for class_name in [
        "0",
        "1"
    ]:

        if class_name in report:

            detailed.append(
                {
                    "model": "BERT",
                    "class": class_name,
                    "precision": report[class_name][
                        "precision"
                    ],
                    "recall": report[class_name][
                        "recall"
                    ],
                    "f1": report[class_name][
                        "f1-score"
                    ],
                    "support": report[class_name][
                        "support"
                    ]
                }
            )

    print(
        "\nAccuracy:",
        round(
            result["accuracy"],
            4
        )
    )

    print(
        "Toxic Precision:",
        round(
            result["precision_toxic"],
            4
        )
    )

    print(
        "Toxic Recall:",
        round(
            result["recall_toxic"],
            4
        )
    )

    print(
        "Toxic F1:",
        round(
            result["f1_toxic"],
            4
        )
    )

    print(
        "Inference Time:",
        round(
            inference_time,
            4
        ),
        "seconds"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion
    )

    return (
        result,
        detailed,
        confusion
    )


# ============================================================
# SAVE FINAL MODEL COMPARISON
# ============================================================

def save_model_comparison(
    results
):

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        results
    )

    path = (
        OUTPUTS_DIR
        / "final_model_comparison.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        "\nSaved:",
        path
    )

    return df


# ============================================================
# SAVE DETAILED CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    reports
):

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        reports
    )

    path = (
        OUTPUTS_DIR
        / "final_classification_report.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    confusion_matrices
):

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        OUTPUTS_DIR
        / "final_confusion_matrices.png"
    )

    names = list(
        confusion_matrices.keys()
    )

    number_of_models = len(
        names
    )

    fig, axes = plt.subplots(
        1,
        number_of_models,
        figsize=(
            5 * number_of_models,
            5
        )
    )

    if number_of_models == 1:

        axes = [axes]

    for axis, name in zip(
        axes,
        names
    ):

        matrix = confusion_matrices[
            name
        ]

        axis.imshow(
            matrix
        )

        axis.set_title(
            name
        )

        axis.set_xlabel(
            "Predicted Label"
        )

        axis.set_ylabel(
            "Actual Label"
        )

        axis.set_xticks(
            [0, 1]
        )

        axis.set_yticks(
            [0, 1]
        )

        axis.set_xticklabels(
            [
                "Non-Toxic",
                "Toxic"
            ]
        )

        axis.set_yticklabels(
            [
                "Non-Toxic",
                "Toxic"
            ]
        )

        for row in range(
            matrix.shape[0]
        ):

            for column in range(
                matrix.shape[1]
            ):

                axis.text(
                    column,
                    row,
                    str(
                        matrix[
                            row,
                            column
                        ]
                    ),
                    ha="center",
                    va="center"
                )

    fig.suptitle(
        "Final Model Confusion Matrices"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    comparison_df,
    test_count
):

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        OUTPUTS_DIR
        / "final_evaluation_summary.txt"
    )

    best_accuracy = comparison_df.loc[
        comparison_df["accuracy"].idxmax()
    ]

    best_f1 = comparison_df.loc[
        comparison_df["f1_toxic"].idxmax()
    ]

    best_recall = comparison_df.loc[
        comparison_df["recall_toxic"].idxmax()
    ]

    fastest = comparison_df.loc[
        comparison_df["inference_time_seconds"].idxmin()
    ]

    lines = []

    lines.append(
        "FINAL MODEL EVALUATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        ""
    )

    lines.append(
        "Test samples: "
        + str(test_count)
    )

    lines.append(
        ""
    )

    lines.append(
        "MODELS EVALUATED"
    )

    lines.append(
        "-" * 70
    )

    for model in comparison_df["model"]:

        lines.append(
            "- "
            + str(model)
        )

    lines.append(
        ""
    )

    lines.append(
        "RESULTS"
    )

    lines.append(
        "-" * 70
    )

    for _, row in comparison_df.iterrows():

        lines.append(
            ""
        )

        lines.append(
            str(row["model"])
        )

        lines.append(
            "Accuracy: "
            + format(
                row["accuracy"],
                ".4f"
            )
        )

        lines.append(
            "Toxic Precision: "
            + format(
                row["precision_toxic"],
                ".4f"
            )
        )

        lines.append(
            "Toxic Recall: "
            + format(
                row["recall_toxic"],
                ".4f"
            )
        )

        lines.append(
            "Toxic F1: "
            + format(
                row["f1_toxic"],
                ".4f"
            )
        )

        lines.append(
            "Inference Time: "
            + format(
                row["inference_time_seconds"],
                ".4f"
            )
            + " seconds"
        )

    lines.append(
        ""
    )

    lines.append(
        "BEST PERFORMERS"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Best Accuracy: "
        + str(
            best_accuracy["model"]
        )
    )

    lines.append(
        "Best Toxic F1: "
        + str(
            best_f1["model"]
        )
    )

    lines.append(
        "Best Toxic Recall: "
        + str(
            best_recall["model"]
        )
    )

    lines.append(
        "Fastest Inference: "
        + str(
            fastest["model"]
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "TEST SET INTEGRITY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "The test text was reconstructed using "
        "the same preprocessing and stratified "
        "train/test split configuration used by "
        "feature_engineering.py."
    )

    lines.append(
        "Random state: 42"
    )

    lines.append(
        "Test size: 20%"
    )

    lines.append(
        "Saved y_test labels were verified against "
        "the reconstructed test labels."
    )

    lines.append(
        ""
    )

    lines.append(
        "Generated by final_evaluation.py"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )

    print(
        "Saved:",
        path
    )


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

def print_final_comparison(
    comparison_df
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    display_df = comparison_df.copy()

    columns = [
        "accuracy",
        "precision_toxic",
        "recall_toxic",
        "f1_toxic",
        "inference_time_seconds"
    ]

    for column in columns:

        display_df[column] = display_df[
            column
        ].round(
            4
        )

    print()

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# VERIFY OUTPUT FILES
# ============================================================

def verify_output_files():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERIFYING FINAL OUTPUTS"
    )

    print(
        "=" * 70
    )

    required_outputs = [
        "final_model_comparison.csv",
        "final_classification_report.csv",
        "final_confusion_matrices.png",
        "final_evaluation_summary.txt"
    ]

    for filename in required_outputs:

        path = (
            OUTPUTS_DIR
            / filename
        )

        verify_file(
            path
        )

        print(
            "OK:",
            filename
        )

    print(
        "\nAll final output files verified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nModels:"
    )

    print(
        "1. Logistic Regression"
    )

    print(
        "2. Linear SVM"
    )

    print(
        "3. BERT"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD SAVED TEST FEATURES / LABELS
    # --------------------------------------------------------

    (
        X_test,
        y_test
    ) = load_saved_test_artifacts()

    # --------------------------------------------------------
    # RECONSTRUCT EXACT TEST TEXT
    # --------------------------------------------------------

    (
        reconstructed_text,
        reconstructed_labels
    ) = load_exact_test_text()

    # --------------------------------------------------------
    # VERIFY EXACT ALIGNMENT
    # --------------------------------------------------------

    (
        test_texts,
        y_test
    ) = verify_test_alignment(
        reconstructed_text,
        reconstructed_labels,
        y_test
    )

    # --------------------------------------------------------
    # LOAD CLASSICAL MODELS
    # --------------------------------------------------------

    (
        ml_models,
        vectorizer
    ) = load_ml_models()

    # --------------------------------------------------------
    # VERIFY ML COMPATIBILITY
    # --------------------------------------------------------

    verify_ml_compatibility(
        ml_models,
        vectorizer,
        X_test
    )

    # --------------------------------------------------------
    # EVALUATE CLASSICAL MODELS
    # --------------------------------------------------------

    all_results = []

    all_reports = []

    all_confusion_matrices = {}

    for name, model in ml_models.items():

        (
            result,
            detailed,
            confusion
        ) = evaluate_classical_model(
            name,
            model,
            X_test,
            y_test
        )

        all_results.append(
            result
        )

        all_reports.extend(
            detailed
        )

        all_confusion_matrices[
            name
        ] = confusion

    # --------------------------------------------------------
    # LOAD BERT
    # --------------------------------------------------------

    (
        tokenizer,
        bert_model,
        device
    ) = load_bert()

    # --------------------------------------------------------
    # EVALUATE BERT
    # --------------------------------------------------------

    (
        bert_result,
        bert_detailed,
        bert_confusion
    ) = evaluate_bert(
        tokenizer,
        bert_model,
        device,
        test_texts,
        y_test
    )

    all_results.append(
        bert_result
    )

    all_reports.extend(
        bert_detailed
    )

    all_confusion_matrices[
        "BERT"
    ] = bert_confusion

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    comparison_df = save_model_comparison(
        all_results
    )

    save_classification_report(
        all_reports
    )

    save_confusion_matrices(
        all_confusion_matrices
    )

    save_summary(
        comparison_df,
        len(y_test)
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print_final_comparison(
        comparison_df
    )

    # --------------------------------------------------------
    # VERIFY OUTPUTS
    # --------------------------------------------------------

    verify_output_files()

    # --------------------------------------------------------
    # FINAL SUCCESS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVALUATION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nAll three models were evaluated on the "
        "same verified test set."
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUTS_DIR
    )

    print(
        "\nGenerated:"
    )

    print(
        "1. final_model_comparison.csv"
    )

    print(
        "2. final_classification_report.csv"
    )

    print(
        "3. final_confusion_matrices.png"
    )

    print(
        "4. final_evaluation_summary.txt"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()