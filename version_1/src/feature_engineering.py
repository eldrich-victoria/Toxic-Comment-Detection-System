import pandas as pd
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# 1. PROJECT PATHS
# -----------------------------

def get_project_root():
    """
    Returns the root directory of the
    Toxic-Comment-Detection-System project.

    Current file:
        Toxic-Comment-Detection-System/
        └── version_1/
            └── src/
                └── feature_engineering.py

    Therefore:
        src -> version_1 -> project root
    """

    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )


def get_version_path():
    """
    Returns the version_1 directory.
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

    project_root = get_project_root()

    processed_path = os.path.join(
        project_root,
        "data",
        "processed"
    )

    input_file = os.path.join(
        processed_path,
        "cleaned_data.csv"
    )

    print("Loading data...")
    print("Input file:", input_file)

    df = pd.read_csv(input_file)

    # Drop rows where clean_text is null
    df = df.dropna(
        subset=["clean_text"]
    )

    return df


# -----------------------------
# 3. TF-IDF FEATURE CREATION
# -----------------------------

def create_tfidf(df):

    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(
        df["clean_text"]
    )

    y = df["target"]

    return X, y, vectorizer


# -----------------------------
# 4. SAVE OBJECTS
# -----------------------------

def save_objects(
    X,
    y,
    vectorizer
):

    version_path = get_version_path()

    # Save model artifacts inside version_1/models
    models_path = os.path.join(
        version_path,
        "models"
    )

    os.makedirs(
        models_path,
        exist_ok=True
    )

    # Save TF-IDF vectorizer
    vectorizer_file = os.path.join(
        models_path,
        "tfidf.pkl"
    )

    with open(
        vectorizer_file,
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file
        )

    # Save features
    X_file = os.path.join(
        models_path,
        "X.pkl"
    )

    with open(
        X_file,
        "wb"
    ) as file:

        pickle.dump(
            X,
            file
        )

    # Save target values
    y_file = os.path.join(
        models_path,
        "y.pkl"
    )

    with open(
        y_file,
        "wb"
    ) as file:

        pickle.dump(
            y,
            file
        )

    print("Saved TF-IDF and data.")
    print("Vectorizer:", vectorizer_file)
    print("Features:", X_file)
    print("Target:", y_file)


# -----------------------------
# 5. MAIN
# -----------------------------

def main():

    print("Loading data...")

    df = load_data()

    print(
        "Data shape:",
        df.shape
    )

    X, y, vectorizer = create_tfidf(
        df
    )

    print(
        "TF-IDF shape:",
        X.shape
    )

    save_objects(
        X,
        y,
        vectorizer
    )

    print(
        "Feature engineering completed successfully."
    )


# -----------------------------
# 6. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()