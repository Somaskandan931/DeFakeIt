# model/train.py

import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bert_lstm_model import BERTLSTMClassifier

# ==== 🔹 Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

DATASET_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/data/isot_fake_news.csv"
MODEL_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/model/model.pt"
TOKENIZER_PATH = "C:/Users/somas/PycharmProjects/DeFakeIt/model/tokenizer.pkl"

# ==== 🔹 Dataset Class ====
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==== 🔹 Load and Split Data ====
def load_data():
    df = pd.read_csv(DATASET_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ==== 🔹 Training Loop ====
def train():
    # Load data and tokenizer
    train_df, val_df = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Save tokenizer
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    # Prepare datasets and loaders
    train_dataset = NewsDataset(train_df.text.tolist(), train_df.label.tolist(), tokenizer)
    val_dataset = NewsDataset(val_df.text.tolist(), val_df.label.tolist(), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model, loss, optimizer
    model = BERTLSTMClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # ==== 🔁 Training Epochs ====
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")
        print(f"   🔹 Train Loss: {total_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # ==== 🔍 Validation ====
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"   ✅ Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print("   💾 Model saved.")

# ==== 🔹 Entry Point ====
if __name__ == "__main__":
    train()
