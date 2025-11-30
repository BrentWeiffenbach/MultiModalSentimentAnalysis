import torch.nn as nn
from src.encoders.bert_encoder import BERTTextEncoder

class BertForMaskedLM(nn.Module):
    def __init__(self, encoder: BERTTextEncoder):
        super().__init__()
        self.bert = encoder
        self.vocab_size = encoder.token_embedding.num_embeddings
        self.hidden_size = encoder.hidden_size
        
        # MLM Head
        # Standard BERT MLM head: Dense -> GeLU -> LayerNorm -> Dense (vocab)
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Weight tying (optional but standard in BERT)
        # This ties the weights of the output layer to the input embeddings
        self.decoder.weight = self.bert.token_embedding.weight
        self.decoder.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] - Labels for computing the masked language modeling loss.
                    Indices should be in [-100, 0, ..., config.vocab_size].
                    Tokens with indices set to -100 are ignored (masked), the loss is only computed for the predictions
                    where the label is not -100.
        """
        # Get sequence output from BERT encoder
        # BERTTextEncoder returns (sentence_repr, token_seq)
        _, sequence_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            return_token_sequence=True
        )
        
        # Pass through MLM head
        # sequence_output: [batch_size, seq_len, hidden_size]
        prediction_scores = self.decoder(self.cls(sequence_output)) # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))
            
        return loss, prediction_scores
