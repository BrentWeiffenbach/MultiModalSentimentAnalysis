# Visual Transformer (ViT) implementation
import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import os
from torch.utils.data import DataLoader
from dataset_mvsa import MVSADataset

def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """Split image into patches and linearly embed them."""
    def __init__(self, img_size: tuple[int, int] | int = 224, patch_size: tuple[int, int] | int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size."
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout) #dropout 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x) #dropout
        x = self.fc2(x)
        x = self.dropout(x) #dropout

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_dropout=attn_drop, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Returns either the class token embedding or the full patch sequence depending on `return_patch_sequence"""
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 representation_size = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 projection_dim = None
                 ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate) #dropout

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, dropout=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # representation layer
        self.representation_size = representation_size
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()

        # classifier head
        self.head = nn.Linear(representation_size or embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # projection head: useful for multimodal alignment (e.g., CLIP-style)
        self.projection_dim = projection_dim
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, projection_dim)
            )
        else:
            self.projection = None

        # init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        nn.init.trunc_normal_(self.cls_token, std = 0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor, return_patch_sequence: bool = False) -> Tuple[Any, torch.Tensor, Optional[Any]]:
        """
        return_patch_sequence: if True return (cls_embedding, patch_sequence_embeddings)
                               else return (cls_embedding, None)
        returns:
            cls_embedding: (B, embed_dim) or projected (B, projection_dim) if projection is set
            patch_seq: (B, N, embed_dim) raw patch tokens after transformer (before pooling), or None
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_embedding = x[:, 0]
        patch_sequence = x[:, 1:]

        cls_rep = self.pre_logits(cls_embedding)

        logits = self.head(cls_rep) if isinstance(self.head, nn.Linear) else None

        if self.projection is not None:
            proj = self.projection(cls_embedding)
            return proj, patch_sequence, logits
        else:
            return cls_rep, patch_sequence, logits
        

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "MVSA", "data")
    labels_file = os.path.join(base_dir, "MVSA", "labelResultAll.txt")
    
    dataset = MVSADataset(data_dir, labels_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    
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
    
    # Test forward pass with batch from dataset
    with torch.no_grad():
        batch = next(iter(loader))
        images = batch["image"].to(device)
        
        print(f"\nInput image shape: {images.shape}")
        
        proj, patches, logits = model(images, return_patch_sequence=True)
        
        print(f"Projection shape: {proj.shape}")
        print(f"Patch sequence shape: {patches.shape}")
        print(f"Logits shape: {logits.shape if logits is not None else 'None'}")
        print(f"Logits (sentiment predictions): {logits}")
        
        print("\n✓ ViT architecture validated successfully!")

if __name__ == "__main__":
    main()


