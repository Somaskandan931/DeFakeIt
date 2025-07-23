import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class BERTLSTMClassifier( nn.Module ) :
    def __init__ ( self, bert_model_name='bert-base-uncased', hidden_dim=128, num_classes=2 ) :
        super( BERTLSTMClassifier, self ).__init__()

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained( bert_model_name )

        # LSTM on top of BERT embeddings
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout( 0.3 )
        self.classifier = nn.Linear( hidden_dim * 2, num_classes )  # *2 for bidirectional

    def forward ( self, input_ids, attention_mask ) :
        # Extract BERT features (frozen)
        with torch.no_grad() :
            outputs = self.bert( input_ids=input_ids, attention_mask=attention_mask )

        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # LSTM processing
        lstm_out, _ = self.lstm( last_hidden_state )  # Shape: (batch_size, seq_len, hidden_dim*2)

        # Mean pooling across sequence length
        pooled = torch.mean( lstm_out, dim=1 )  # Shape: (batch_size, hidden_dim*2)

        # Classification
        dropped = self.dropout( pooled )
        logits = self.classifier( dropped )

        return logits


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERTLSTMClassifier()
model.load_state_dict(torch.load("C:/Users/somas/PycharmProjects/DeFakeIt/model/model.pt", map_location=torch.device('cpu')))
model.eval()

def predict_news(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        probs = torch.softmax(logits, dim=1).squeeze()
    label = "fake" if probs[1] > 0.5 else "real"
    return {"label": label, "fake_probability": float(probs[1])}
