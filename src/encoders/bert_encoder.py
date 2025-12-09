from typing import Optional, Tuple
import torch
import torch.nn as nn


class BERTTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        
        self.embed_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_dim = hidden_size
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_token_sequence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)
        
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        
        # Create attention mask for transformer (inverted: 0 = attend, 1 = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # CLS token pooling (first token)
        pooled = encoded[:, 0]
        
        sentence_repr = pooled
                
        token_seq = encoded if return_token_sequence else None
        
        return sentence_repr, token_seq