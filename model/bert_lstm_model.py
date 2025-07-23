# model/bert_lstm_model.py

import torch
import torch.nn as nn
from transformers import BertModel


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
