import pandas as pd
import re


# Import combined dataset
df = pd.read_csv("data/processed/combined_data.csv")


def clean_text_advanced(text):
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (#word → word)
    text = re.sub(r"#", "", text)

    # Remove emojis (basic)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text

print("Applying advanced text cleaning...")
df["clean_text"] = df["text"].apply(clean_text_advanced)



df.to_csv("data/processed/cleaned_data.csv", index=False)

df[['text', 'clean_text']].sample(5)


print(df['clean_text'].str.len().describe())
import pandas as pd
import re
import os


# -----------------------------
# 1. PROJECT PATH
# -----------------------------

def get_project_root():
    """
    Returns the root directory of the
    Toxic-Comment-Detection-System project.

    Current file:
        Toxic-Comment-Detection-System/
        └── version_1/
            └── src/
                └── data_preprocessing_2.py

    Therefore, we go three levels up:
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
# 2. LOAD COMBINED DATASET
# -----------------------------

def load_dataset():

    project_root = get_project_root()

    processed_path = os.path.join(
        project_root,
        "data",
        "processed"
    )

    input_file = os.path.join(
        processed_path,
        "combined_data.csv"
    )

    print("Loading combined dataset...")
    print("Input file:", input_file)

    df = pd.read_csv(input_file)

    return df, processed_path


# -----------------------------
# 3. ADVANCED TEXT CLEANING
# -----------------------------

def clean_text_advanced(text):

    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtags (#word → word)
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove emojis and special characters
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove extra spaces
    text = " ".join(
        text.split()
    )

    return text


# -----------------------------
# 4. MAIN FUNCTION
# -----------------------------

def main():

    # Load combined dataset
    df, processed_path = load_dataset()

    print("Applying advanced text cleaning...")

    # Apply cleaning
    df["clean_text"] = df["text"].apply(
        clean_text_advanced
    )

    # -----------------------------
    # 5. SAVE CLEANED DATASET
    # -----------------------------

    output_file = os.path.join(
        processed_path,
        "cleaned_data.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "Cleaned dataset saved to:",
        output_file
    )

    # -----------------------------
    # 6. DISPLAY SAMPLE
    # -----------------------------

    print("\nSample cleaned data:")

    print(
        df[
            [
                "text",
                "clean_text"
            ]
        ].sample(5)
    )

    # -----------------------------
    # 7. TEXT LENGTH STATISTICS
    # -----------------------------

    print("\nClean text length statistics:")

    print(
        df["clean_text"]
        .str.len()
        .describe()
    )

    print(
        "\nAdvanced text cleaning completed successfully."
    )


# -----------------------------
# 8. RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    main()