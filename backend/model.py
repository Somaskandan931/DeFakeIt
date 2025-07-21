import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertLSTMClassifier()
model.load_state_dict(torch.load("model/model.pt", map_location=torch.device('cpu')))
model.eval()

def predict_news(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        probs = torch.softmax(logits, dim=1).squeeze()
    label = "fake" if probs[1] > 0.5 else "real"
    return {"label": label, "fake_probability": float(probs[1])}
