# backend/shap_explainer.py

import torch
import shap
from backend.model import model, tokenizer

# Ensure model is in evaluation mode and on correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define a function to return probabilities from model
def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# Initialize the SHAP explainer (only once)
explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))

# Main explanation function
def explain_text(text: str):
    shap_values = explainer([text])
    return shap_values
