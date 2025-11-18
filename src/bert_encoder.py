# BERT text encoding

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from dataset_mvsa import MVSADataset
import os
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel

Device = torch.device

def cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
   
    return last_hidden[:, 0] 

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
 
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  
    summed = (last_hidden * mask).sum(dim=1)                   
    denom = mask.sum(dim=1).clamp(min=1e-6)                    
    return summed / denom

class BERTTextEncoder(nn.Module):
    """
    Returns:
      - sentence_repr: (B, hidden) or projected (B, projection_dim) if projection set
      - token_sequence: (B, T, H) token embeddings after the last transformer layer
      - logits: (B, num_classes) if classifier head is enabled, else None
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "cls",             
        num_classes: int = 0,            
        projection_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.pooling = pooling.lower()
        assert self.pooling in {"cls", "mean"}


        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, projection_dim)
            )
            self.out_dim = projection_dim
        else:
            self.out_dim = self.hidden_size

        self.classifier = None
        if num_classes and num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.out_dim, num_classes)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_token_sequence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
          input_ids, attention_mask, token_type_ids: standard HF tensors [B, T]
        Returns:
          sentence_repr, token_seq (or None), logits (or None)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state   

        if self.pooling == "cls":
            pooled = cls_pool(last_hidden)  
        else:
            pooled = mean_pool(last_hidden, attention_mask)  

        if self.projection is not None:
            sentence_repr = self.projection(pooled)           
        else:
            sentence_repr = pooled                           

        logits = None
        if self.classifier is not None:
            logits = self.classifier(sentence_repr)           

        token_seq = last_hidden if return_token_sequence else None
        return sentence_repr, token_seq, logits


@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor]
    texts: List[str]

class TextCollator:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> TextBatch:
        texts = [str(x["text"]) for x in examples]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        token_type_ids = enc.get("token_type_ids", None)
        return TextBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=token_type_ids,
            texts=texts
        )

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mvsa_dir = os.path.abspath(os.path.join(base_dir, "..", "MVSA"))
    data_dir = os.path.join(mvsa_dir, "data")
    labels_file = os.path.join(mvsa_dir, "labelResultAll.txt")

    dataset = MVSADataset(data_dir, labels_file)

    collator = TextCollator(model_name="bert-base-uncased")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    print(f"Dataset size: {len(dataset)}")

    model = BERTTextEncoder(
        projection_dim=256,
        num_classes=3
    ).to(device)

    model.eval()
    print("BERT model initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

   
    with torch.no_grad():
        batch = next(iter(loader))
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        token_type_ids = (
            batch.token_type_ids.to(device)
            if batch.token_type_ids is not None else None
        )

        # print(f"\nExample texts: {batch.texts[:2]}")

        
        sentence_repr, token_seq, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")

        print(f"Token sequence shape: {token_seq.shape}")
        print(f"Sentence representation shape: {sentence_repr.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits: {logits}")

        print("\n✓ BERT text encoder validated successfully on MVSA!")

if __name__ == "__main__":
    main()
