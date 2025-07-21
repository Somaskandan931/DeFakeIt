import pandas as pd
from pymongo import MongoClient
from model.bert_lstm_model import tokenizer, FakeNewsDataset, BertLSTMClassifier, MAX_LEN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

# ========== 🔹 MongoDB ==========
client = MongoClient("mongodb://localhost:27017/")
db = client["fake_news_detection"]
live_news = db["verified_articles"]

# ========== 🔹 Load ISOT Dataset ==========
isot_df = pd.read_csv("data/isot_fake_news.csv")
isot_df = isot_df[["text", "label"]].dropna()
isot_df["label"] = isot_df["label"].map({"real": 0, "fake": 1})

# ========== 🔹 Load Live News ==========
live_docs = list(live_news.find({"label": {"$in": ["real", "fake"]}}))
live_df = pd.DataFrame(live_docs)

if not live_df.empty:
    live_df = live_df[["title", "label"]]
    live_df.columns = ["text", "label"]
    live_df["label"] = live_df["label"].map({"real": 0, "fake": 1})
    combined_df = pd.concat([isot_df, live_df], ignore_index=True)
else:
    print("⚠️ No manually labeled live articles found. Skipping retrain.")
    combined_df = isot_df

# ========== 🔹 Training ==========
dataset = FakeNewsDataset(combined_df["text"].tolist(), combined_df["label"].tolist(), tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BertLSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.train()
for epoch in range(2):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Retraining Epoch {epoch+1}/2"):
        optimizer.zero_grad()
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model/model.pt")
print("✅ Retrained model saved.")
