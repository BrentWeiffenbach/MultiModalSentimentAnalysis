import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import os
from torch.utils.data import DataLoader
from dataset_mvsa import MVSADataset
from VisionTransformer import VisionTransformer
from bert_text_encoder import BertTextEncoder

class ClassificationHead(nn.Module):
    """Classification head for output of multi-modal feature fusion."""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.5):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

def main():
    # test usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "MVSA", "data")
    labels_file = os.path.join(base_dir, "MVSA", "labelResultAll.txt")
    
    dataset = MVSADataset(data_dir, labels_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=3,
        embed_dim=768,
        depth=4,
        num_heads=8,
        projection_dim=256
    )
    model.to(device)
    model.eval()

    print("ViT Model initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    text_encoder = BertTextEncoder(
        pretrained_model_name='bert-base-uncased',
        output_dim=256
    )
    text_encoder.to(device)
    text_encoder.eval()

    print("BERT Text Encoder initialized successfully")
    print(f"Text Encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")

    with torch.no_grad():
        batch = next(iter(loader))
        images = batch["image"].to(device)
        texts = batch["text"]

        print(f"\nInput image shape: {images.shape}")
        print(f"Input text batch size: {len(texts)}")

        proj, patches, logits = model(images, return_patch_sequence=True)
        print(f"Projection shape: {proj.shape}")
        print(f"Patch sequence shape: {patches.shape}")
        print(f"Logits shape: {logits.shape if logits is not None else 'None'}")
        print(f"Logits (sentiment predictions): {logits}")

        text_features = text_encoder(texts, device=device)
        print(f"Text feature shape: {text_features.shape}")

        fused_features = torch.cat([proj, text_features], dim=1)
        print(f"Fused feature shape: {fused_features.shape}")

        classification_head = ClassificationHead(
            input_dim=fused_features.shape[1],
            num_classes=3
        ).to(device)
        classification_head.eval()

        output_logits = classification_head(fused_features)
        print(f"Classification logits shape: {output_logits.shape}")
        print(f"Classification logits: {output_logits}")

        print("\n✓ Multi-modal feature fusion and classification validated successfully!")