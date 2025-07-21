# model/evaluate.py
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer
from bert_lstm_model import BERTLSTMClassifier
import pickle

# Load test data
df = pd.read_csv("data/test.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Tokenize
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTLSTMClassifier()
model.load_state_dict(torch.load("model/model.pt", map_location=device))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    outputs = model(input_ids.to(device), attention_mask.to(device))
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# Report
print("✅ Classification Report:\n")
print(classification_report(labels, preds, target_names=["Real", "Fake"]))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(labels, preds))
