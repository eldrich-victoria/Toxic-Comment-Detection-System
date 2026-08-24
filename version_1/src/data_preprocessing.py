import pandas as pd
import os


# -----------------------------
# 1. PROJECT PATH
# -----------------------------

def get_project_root():
    """
    Returns the root directory of the Toxic-Comment-Detection-System project.

    Current file:
        Toxic-Comment-Detection-System/
        └── version_1/
            └── src/
                └── data_preprocessing.py

    Therefore, we go three levels up from this file:
        src -> version_1 -> project root
    """

    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )


# -----------------------------
# 2. LOAD DATASETS
# -----------------------------

def load_datasets():

    project_root = get_project_root()

    # Path to raw datasets
    data_path = os.path.join(
        project_root,
        "data",
        "raw"
    )

    print("Raw data path:", data_path)

    # Jigsaw dataset
    jigsaw = pd.read_csv(
        os.path.join(
            data_path,
            "Jigsaw_train.csv"
        )
    )

    # Davidson dataset
    davidson = pd.read_csv(
        os.path.join(
            data_path,
            "davidson_train.csv"
        )
    )

    # Unintended Bias dataset
    bias = pd.read_csv(
        os.path.join(
            data_path,
            "Unintended_train.csv"
        )
    )

    return jigsaw, davidson, bias


# -----------------------------
# 3. STANDARDIZE JIGSAW
# -----------------------------

def process_jigsaw(df):

    df["target"] = (
        df["toxic"]
        | df["severe_toxic"]
        | df["insult"]
        | df["threat"]
        | df["identity_hate"]
    ).astype(int)

    df = df[
        [
            "comment_text",
            "target"
        ]
    ]

    df.columns = [
        "text",
        "target"
    ]

    return df


# -----------------------------
# 4. STANDARDIZE DAVIDSON
# -----------------------------

def process_davidson(df):

    # 0 = hate, 1 = offensive → toxic
    # 2 = neither → clean

    df["target"] = df["class"].apply(
        lambda x: 0 if x == 2 else 1
    )

    df = df[
        [
            "tweet",
            "target"
        ]
    ]

    df.columns = [
        "text",
        "target"
    ]

    return df


# -----------------------------
# 5. STANDARDIZE BIAS DATASET
# -----------------------------

def process_bias(df):

    # Convert continuous toxicity → binary

    df["target"] = (
        df["target"] >= 0.5
    ).astype(int)

    df = df[
        [
            "comment_text",
            "target"
        ]
    ]

    df.columns = [
        "text",
        "target"
    ]

    return df


# -----------------------------
# 6. MERGE DATA
# -----------------------------

def merge_datasets(
    jigsaw,
    davidson,
    bias
):

    df = pd.concat(
        [
            jigsaw,
            davidson,
            bias
        ],
        ignore_index=True
    )

    # Remove duplicates
    df.drop_duplicates(
        inplace=True
    )

    # Reset index
    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# -----------------------------
# 7. NORMALIZE TEXT
# -----------------------------

def normalize_text(text):

    if pd.isna(text):
        return ""

    # Remove surrounding quotes
    text = text.strip('"').strip("'")

    # Remove leading/trailing spaces
    text = text.strip()

    # Replace multiple spaces with a single space
    text = " ".join(
        text.split()
    )

    return text


# -----------------------------
# 8. SAVE PROCESSED DATA
# -----------------------------

def save_processed_data(df):

    project_root = get_project_root()

    # Path to processed data directory
    processed_path = os.path.join(
        project_root,
        "data",
        "processed"
    )

    # Create directory if it does not exist
    os.makedirs(
        processed_path,
        exist_ok=True
    )

    # Output file
    output_file = os.path.join(
        processed_path,
        "combined_data.csv"
    )

    # Save dataset
    df.to_csv(
        output_file,
        index=False
    )

    print(
        "Saved to:",
        output_file
    )


# -----------------------------
# 9. MAIN FUNCTION
# -----------------------------

def main():

    print("Loading datasets...")

    jigsaw, davidson, bias = load_datasets()

    print("Processing datasets...")

    jigsaw = process_jigsaw(jigsaw)

    davidson = process_davidson(davidson)

    bias = process_bias(bias)

    print("Merging datasets...")

    df = merge_datasets(
        jigsaw,
        davidson,
        bias
    )

    print(
        "Final dataset shape:",
        df.shape
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["target"].value_counts()
    )

    print(
        "\nCleaning text formatting..."
    )

    df["text"] = df["text"].apply(
        normalize_text
    )

    print(
        "\nSaving processed dataset..."
    )

    save_processed_data(df)

    print(
        "\nData preprocessing completed successfully."
    )


# -----------------------------
# 10. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()