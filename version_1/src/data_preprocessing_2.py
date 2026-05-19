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
