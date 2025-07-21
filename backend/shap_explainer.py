import shap
import torch
from backend.model import model, tokenizer

# Make sure the model is in eval mode and on CPU (or CUDA if available)
model.eval()

def get_shap_explanation(text):
    # Tokenize input text with padding/truncation
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Define a prediction function compatible with SHAP
    def model_forward(input_ids_attention_mask):
        input_ids = torch.tensor(input_ids_attention_mask[0]).to(next(model.parameters()).device)
        attention_mask = torch.tensor(input_ids_attention_mask[1]).to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    # SHAP expects input as a list of arrays (for each input arg to model_forward)
    explainer = shap.Explainer(model_forward, masker=shap.maskers.Text(tokenizer))

    # Prepare input as a tuple/list matching model_forward inputs
    # Since tokenizer returns dict of tensors, extract to lists for SHAP
    input_for_shap = [input_ids[0].tolist(), attention_mask[0].tolist()]

    # Get SHAP values for the single input
    shap_values = explainer([input_for_shap])

    return shap_values
