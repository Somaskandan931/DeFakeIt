import torch
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from bert_lstm_model import BERTLSTMClassifier  # Ensure this file is in your working directory or PYTHONPATH

# Set paths (adjust as needed)
DATA_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/data/isot_fake_news.csv"
TOKENIZER_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/model/tokenizer.pkl"
MODEL_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/model/model.pt"

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load dataset and create validation split
df = pd.read_csv(DATA_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
texts = val_df["text"].tolist()
labels = val_df["label"].tolist()

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# To reduce GPU memory usage, reduce max_length if needed (e.g., 128 or 256)
MAX_LEN = 256

# Tokenize texts with padding and truncation
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)

# Move tensors to device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Load labels as tensor on CPU (for sklearn compatibility)
labels_tensor = torch.tensor(labels)  # Keep on CPU

# Load model and move to device
model = BERTLSTMClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Free any cached memory before inference (optional but helpful)
torch.cuda.empty_cache()

# Inference: disable gradient calculations
with torch.no_grad():
    outputs = model(input_ids, attention_mask)

    # Assuming outputs are logits for 2 classes
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# Print classification report
print("✅ Classification Report:\n")
print(classification_report(labels, preds, target_names=["Real", "Fake"]))

# Plot confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
