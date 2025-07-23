# model/predict.py
import torch
import pickle
from transformers import BertTokenizer
from model.bert_lstm_model import BERTLSTMClassifier

# Load tokenizer
with open("C:/Users/somas/PycharmProjects/DeFakeIt/model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTLSTMClassifier()
model.load_state_dict(torch.load("C:/Users/somas/PycharmProjects/DeFakeIt/model/model.pt", map_location=device))
model.to(device)
model.eval()

def predict_news(text: str):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = "fake" if pred == 1 else "real"
        return {
            "label": label,
            "fake_probability": round(probs[0][1].item(), 4),
            "real_probability": round(probs[0][0].item(), 4)
        }
