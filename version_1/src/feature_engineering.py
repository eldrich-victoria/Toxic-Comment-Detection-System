import pandas as pd
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    # Drop rows where clean_text is null
    df = df.dropna(subset=["clean_text"])
    return df


# -----------------------------
# TF-IDF FEATURE CREATION
# -----------------------------

def create_tfidf(df):
    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["target"]

    return X, y, vectorizer


# -----------------------------
# SAVE OBJECTS
# -----------------------------

def save_objects(X, y, vectorizer):
    os.makedirs("models", exist_ok=True)

    # Save vectorizer
    pickle.dump(vectorizer, open("models/tfidf.pkl", "wb"))

    # Save features (optional but useful)
    pickle.dump(X, open("models/X.pkl", "wb"))
    pickle.dump(y, open("models/y.pkl", "wb"))

    print("Saved TF-IDF and data.")


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("Loading data...")
    df = load_data()

    print("Data shape:", df.shape)

    X, y, vectorizer = create_tfidf(df)

    print("TF-IDF shape:", X.shape)

    save_objects(X, y, vectorizer)


if __name__ == "__main__":
    main()