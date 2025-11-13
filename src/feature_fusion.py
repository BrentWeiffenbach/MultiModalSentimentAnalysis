# Multimodal Feature Fusion Module
import torch
import torch.nn as nn
from attention import Attention
from mlp import MLP

class FusionBlock(nn.Module):
    """Transformer block with multi-head attention and MLP, both with residual connections and layer normalization."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True, dropout: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_dropout=attn_drop, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection and layer norm
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection and layer norm
        x = x + self.mlp(self.norm2(x))

        return x


class MultimodalFeatureFusion(nn.Module):
    """
    Multimodal feature fusion module that combines BERT text features and ViT vision features.
    """
    def __init__(self,
                 bert_dim: int = 768,
                 vit_dim: int = 768,
                 fusion_dim: int = 768,
                 num_heads: int = 8,
                 depth: int = 2,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 attn_drop: float = 0.0,
                 max_seq_len: int = 512
                 ):
        super().__init__()
        
        self.bert_dim = bert_dim
        self.vit_dim = vit_dim
        self.fusion_dim = fusion_dim
        self.max_seq_len = max_seq_len
        
        # Project BERT and ViT features to fusion dimension if needed
        self.bert_proj = nn.Linear(bert_dim, fusion_dim) if bert_dim != fusion_dim else nn.Identity()
        self.vit_proj = nn.Linear(vit_dim, fusion_dim) if vit_dim != fusion_dim else nn.Identity()
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, fusion_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.blocks = nn.ModuleList([
            FusionBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_drop=attn_drop
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(fusion_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard transformer initialization."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self, bert_features: torch.Tensor, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multimodal fusion module.
        """
        # Project features to fusion dimension
        bert_features = self.bert_proj(bert_features)  # (B, N_text, fusion_dim)
        vit_features = self.vit_proj(vit_features)      # (B, N_vision, fusion_dim)
        
        # Concatenate BERT and ViT features
        x = torch.cat([bert_features, vit_features], dim=1)
        N = x.shape[1]
        
        assert N <= self.max_seq_len, f"Sequence length {N} exceeds max_seq_len {self.max_seq_len}"
        
        # Add positional encoding
        x = x + self.pos_embed[:, :N, :]
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


def main():
    """Test the multimodal feature fusion module."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize fusion module
    fusion = MultimodalFeatureFusion(
        bert_dim=768,
        vit_dim=768,
        fusion_dim=768,
        num_heads=8,
        depth=2,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_drop=0.1,
        max_seq_len=512
    )
    fusion.to(device)
    fusion.eval()
    
    print("Multimodal Feature Fusion Module initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    
    batch_size = 2
    text_seq_len = 128
    vision_seq_len = 196
    
    # Create dummy inputs
    bert_features = torch.randn(batch_size, text_seq_len, 768).to(device)
    vit_features = torch.randn(batch_size, vision_seq_len, 768).to(device)
    
    print("\nInput shapes:")
    print(f"  BERT features: {bert_features.shape}")
    print(f"  ViT features: {vit_features.shape}")
    
    # Forward pass
    with torch.no_grad():
        fused_features = fusion(bert_features, vit_features)
    
    print("\nOutput shape:")
    print(f"  Fused features: {fused_features.shape}")
    print(f"  Expected: ({batch_size}, {text_seq_len + vision_seq_len}, 768)")
    
    print("\n✓ Multimodal feature fusion validated successfully!")


if __name__ == "__main__":
    main()
