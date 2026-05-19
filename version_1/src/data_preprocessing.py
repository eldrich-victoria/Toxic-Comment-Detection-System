import pandas as pd
import os

# -----------------------------
# 1. LOAD DATASETS
# -----------------------------

def load_datasets():
    data_path = "data/raw/"

    # Jigsaw dataset
    jigsaw = pd.read_csv(os.path.join(data_path, "Jigsaw_train.csv"))

    # Davidson dataset
    davidson = pd.read_csv(os.path.join(data_path, "davidson_train.csv"))

    # Unintended bias dataset
    bias = pd.read_csv(os.path.join(data_path, "Unintended_train.csv"))  # rename if needed

    return jigsaw, davidson, bias


# -----------------------------
# 2. STANDARDIZE JIGSAW
# -----------------------------

def process_jigsaw(df):
    df["target"] = (
        df["toxic"] |
        df["severe_toxic"] |
        df["insult"] |
        df["threat"] |
        df["identity_hate"]
    ).astype(int)

    df = df[["comment_text", "target"]]
    df.columns = ["text", "target"]

    return df


# -----------------------------
# 3. STANDARDIZE DAVIDSON
# -----------------------------

def process_davidson(df):
    # 0 = hate, 1 = offensive → toxic
    # 2 = neither → clean
    df["target"] = df["class"].apply(lambda x: 0 if x == 2 else 1)

    df = df[["tweet", "target"]]
    df.columns = ["text", "target"]

    return df


# -----------------------------
# 4. STANDARDIZE BIAS DATASET
# -----------------------------

def process_bias(df):
    # Convert continuous toxicity → binary
    df["target"] = (df["target"] >= 0.5).astype(int)

    df = df[["comment_text", "target"]]
    df.columns = ["text", "target"]

    return df


# -----------------------------
# 5. MERGE DATA
# -----------------------------

def merge_datasets(jigsaw, davidson, bias):
    df = pd.concat([jigsaw, davidson, bias], ignore_index=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# -----------------------------
# 6. NORMALIZE TEXT FUNCTION
# -----------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    # Remove surrounding quotes
    text = text.strip('"').strip("'")

    # Remove leading/trailing spaces
    text = text.strip()

    # Replace multiple spaces with single space
    text = " ".join(text.split())

    return text

# -----------------------------
# 7. MAIN FUNCTION
# -----------------------------

def main():
    print("Loading datasets...")
    jigsaw, davidson, bias = load_datasets()

    print("Processing datasets...")
    jigsaw = process_jigsaw(jigsaw)
    davidson = process_davidson(davidson)
    bias = process_bias(bias)

    print("Merging datasets...")
    df = merge_datasets(jigsaw, davidson, bias)

    print("Final dataset shape:", df.shape)
    print(df["target"].value_counts())

    print("Cleaning text formatting...")
    df["text"] = df["text"].apply(normalize_text)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/combined_data.csv", index=False)

    print("Saved to data/processed/combined_data.csv")


if __name__ == "__main__":
    main()