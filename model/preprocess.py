import pandas as pd
import os
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===== Manually set NLTK data path =====
NLTK_PATH = "C:/Users/somas/AppData/Roaming/nltk_data"
nltk.data.path.clear()
nltk.data.path.append(NLTK_PATH)

# ===== Validate NLTK resources =====
stopwords_path = os.path.join(NLTK_PATH, "corpora", "stopwords", "english")
punkt_path = os.path.join(NLTK_PATH, "tokenizers", "punkt", "english.pickle")

if not os.path.exists(stopwords_path):
    raise RuntimeError(f"❌ 'stopwords' not found at: {stopwords_path}")

if not os.path.exists(punkt_path):
    raise RuntimeError(f"❌ 'punkt' tokenizer not found at: {punkt_path}")

# ===== Load NLTK Resources =====
STOPWORDS = set(stopwords.words("english"))

# ===== Text Cleaning =====
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# ===== Preprocess ISOT Dataset =====
def preprocess_isot(
    true_path="C:/Users/somas/PycharmProjects/DeFakeIt/data/True.csv",
    fake_path="C:/Users/somas/PycharmProjects/DeFakeIt/data/Fake.csv",
    output_csv="C:/Users/somas/PycharmProjects/DeFakeIt/data/isot_fake_news.csv",
    output_json="C:/Users/somas/PycharmProjects/DeFakeIt/data/processed.json"
):
    df_real = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_real["label"] = 0
    df_fake["label"] = 1

    df = pd.concat([df_real, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["text"] = df["text"].apply(clean_text)

    df = df[["text", "label"]].dropna()
    df = df[df["text"].str.strip().astype(bool)]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessed CSV saved to: {output_csv} ({len(df)} samples)")

    df.to_json(output_json, orient="records", lines=True)
    print(f"✅ Preprocessed JSONL saved to: {output_json}")

# ===== Script Entry =====
if __name__ == "__main__":
    preprocess_isot()
