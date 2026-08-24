import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATH CONFIGURATION
# ============================================================

# This file is expected at:
# D:\Toxic-Comment-Detection-System\version_2\src\feature_engineering.py

VERSION_2_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = VERSION_2_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

MODELS_DIR = VERSION_2_DIR / "models"


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_FEATURES = 100000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
SUBLINEAR_TF = True


# ============================================================
# REQUIRED OUTPUT FILES
# ============================================================

X_TRAIN_PATH = MODELS_DIR / "X_train.pkl"
X_TEST_PATH = MODELS_DIR / "X_test.pkl"

Y_TRAIN_PATH = MODELS_DIR / "y_train.pkl"
Y_TEST_PATH = MODELS_DIR / "y_test.pkl"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_pickle_safely(obj, destination):
    """
    Save an object using a temporary file and replace the
    destination only after the temporary file is written
    successfully.
    """

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=destination.stem + "_",
        suffix=".tmp"
    )

    os.close(file_descriptor)

    try:
        with open(
            temporary_path,
            "wb"
        ) as file:

            pickle.dump(
                obj,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(
            temporary_path,
            destination
        )

    except Exception:

        if os.path.exists(temporary_path):
            os.remove(temporary_path)

        raise


def verify_file(path):
    """
    Verify that an output file exists and is not empty.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Expected artifact was not created:\n"
            + str(path)
        )

    if path.stat().st_size == 0:
        raise ValueError(
            "Artifact was created but is empty:\n"
            + str(path)
        )


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("\nLoading source dataset...")
    print("Dataset path:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Source dataset not found:\n"
            + str(DATA_PATH)
        )

    df = pd.read_csv(DATA_PATH)

    print("\nOriginal dataset shape:")
    print(df.shape)

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
            "Required columns are missing: "
            + str(missing_columns)
        )

    df = df[
        [
            "clean_text",
            "target"
        ]
    ].copy()

    # Clean text
    df["clean_text"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    # Clean target
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

    # Keep only binary target classes
    df = df[
        df["target"].isin([0, 1])
    ]

    # Remove duplicate comments
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["clean_text", "target"]
    )

    removed_duplicates = (
        before_duplicates
        - len(df)
    )

    print("\nAfter cleaning:")
    print(df.shape)

    print(
        "Duplicate rows removed:",
        removed_duplicates
    )

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    if len(df) == 0:
        raise ValueError(
            "No usable data remains after preprocessing."
        )

    if df["target"].nunique() < 2:
        raise ValueError(
            "Target contains fewer than two classes."
        )

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(df):

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X = df["clean_text"]
    y = df["target"]

    print("\nPerforming stratified 80/20 split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:")
    print(len(X_train))

    print("Testing samples:")
    print(len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TF-IDF FEATURE ENGINEERING
# ============================================================

def create_tfidf_features(
    X_train,
    X_test
):

    print("\n" + "=" * 60)
    print("TF-IDF FEATURE ENGINEERING")
    print("=" * 60)

    print("\nCreating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF
    )

    print("\nFitting TF-IDF on training data only...")

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    print("Transforming test data...")

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print("\nTF-IDF feature generation completed.")

    print(
        "X_train TF-IDF shape:",
        X_train_tfidf.shape
    )

    print(
        "X_test TF-IDF shape:",
        X_test_tfidf.shape
    )

    print(
        "Number of vocabulary features:",
        len(vectorizer.vocabulary_)
    )

    return (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer
):

    print("\n" + "=" * 60)
    print("SAVING FEATURE ARTIFACTS")
    print("=" * 60)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    artifacts = {
        X_TRAIN_PATH: X_train,
        X_TEST_PATH: X_test,
        Y_TRAIN_PATH: y_train,
        Y_TEST_PATH: y_test,
        VECTORIZER_PATH: vectorizer
    }

    print("\nSaving artifacts...")

    for path, obj in artifacts.items():

        print(
            "Saving:",
            path
        )

        save_pickle_safely(
            obj,
            path
        )

        verify_file(path)

        print(
            "Verified:",
            path
        )

    print("\nAll feature artifacts saved successfully.")


# ============================================================
# VERIFY SAVED ARTIFACTS
# ============================================================

def verify_artifacts():

    print("\n" + "=" * 60)
    print("ARTIFACT VERIFICATION")
    print("=" * 60)

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        VECTORIZER_PATH
    ]

    for path in required_files:

        verify_file(path)

        print(
            "OK:",
            path
        )

    print("\nLoading artifacts for integrity check...")

    with open(
        X_TRAIN_PATH,
        "rb"
    ) as file:
        X_train = pickle.load(file)

    with open(
        X_TEST_PATH,
        "rb"
    ) as file:
        X_test = pickle.load(file)

    with open(
        Y_TRAIN_PATH,
        "rb"
    ) as file:
        y_train = pickle.load(file)

    with open(
        Y_TEST_PATH,
        "rb"
    ) as file:
        y_test = pickle.load(file)

    with open(
        VECTORIZER_PATH,
        "rb"
    ) as file:
        vectorizer = pickle.load(file)

    print(
        "\nX_train shape:",
        X_train.shape
    )

    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "y_train length:",
        len(y_train)
    )

    print(
        "y_test length:",
        len(y_test)
    )

    print(
        "Vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "X_train and y_train sizes do not match."
        )

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "X_test and y_test sizes do not match."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "X_train and X_test have different feature dimensions."
        )

    print("\nArtifact integrity verification PASSED.")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    ) = create_tfidf_features(
        X_train,
        X_test
    )

    save_features(
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    )

    verify_artifacts()

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nThe following artifacts are ready:")
    print(
        "1.",
        X_TRAIN_PATH
    )
    print(
        "2.",
        X_TEST_PATH
    )
    print(
        "3.",
        Y_TRAIN_PATH
    )
    print(
        "4.",
        Y_TEST_PATH
    )
    print(
        "5.",
        VECTORIZER_PATH
    )

    print(
        "\nNext stage: model_training.py"
    )

    print(
        "Do NOT run the next stage until this script "
        "finishes with 'ARTIFACT INTEGRITY VERIFICATION PASSED'."
    )


if __name__ == "__main__":
    main()