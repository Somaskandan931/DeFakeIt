import re
import string

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove digits
    text = text.strip()
    return text
