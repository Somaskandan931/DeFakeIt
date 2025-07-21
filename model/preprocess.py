import pandas as pd
import os
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data once
nltk.download("punkt")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Preprocess raw news text."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)               # Remove URLs
    text = re.sub(r"\d+", "", text)                   # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def preprocess_isot(
    true_path: str = "C:/Users/somas/PycharmProjects/DeFakeIt/data/True.csv",
    fake_path: str = "C:/Users/somas/PycharmProjects/DeFakeIt/data/Fake.csv",
    output_csv: str = "C:/Users/somas/PycharmProjects/DeFakeIt/data/isot_fake_news.csv",
    output_json: str = "C:/Users/somas/PycharmProjects/DeFakeIt/data/processed.json"
):
    # Load raw CSV files
    df_real = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Add label: 0 = real, 1 = fake
    df_real["label"] = 0
    df_fake["label"] = 1

    # Merge and shuffle
    df = pd.concat([df_real, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop rows with empty text after cleaning
    df = df[["text", "label"]].dropna()
    df = df[df["text"].str.strip().astype(bool)]

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessed dataset saved to {output_csv} ({len(df)} samples)")

    # Save as JSONL for flexibility (optional)
    df.to_json(output_json, orient="records", lines=True)
    print(f"✅ JSONL version saved to {output_json}")

if __name__ == "__main__":
    preprocess_isot()
