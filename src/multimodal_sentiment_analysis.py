import torch
import torch.nn as nn
from src.classification_head import ClassificationHead
from src.feature_fusion import MultimodalFeatureFusion
from src.encoders.vision_transformer import VisionTransformer
from src.encoders.bert_encoder import BERTTextEncoder
from typing import Optional
from transformers import BertModel, ViTModel

class MultiModalSentimentAnalysis(nn.Module):
    """
    Multi-modal sentiment analysis model using ViT for images and BERT for text.
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.2, freeze_encoders: bool = False, 
                 pretrained_vit_path: Optional[str] = None, pretrained_bert_path: Optional[str] = None,
                 use_huggingface: bool = False, use_fusion: bool = False):
        super(MultiModalSentimentAnalysis, self).__init__()
        
        self.use_huggingface = use_huggingface
        self.use_fusion = use_fusion
        # fusion_feature_dim = 512

        if use_huggingface:
            print("Using HuggingFace pre-trained models (bert-base-uncased, google/vit-base-patch16-224)")
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
            
            self.bert_hidden_size = 768
            self.vit_hidden_size = 768
        else:
            # Load custom pre-trained models
            # Updated to match pre-training configs (ViT-Small and BERT-Medium)
            embedding_dim = 384
            
            self.vit = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=0,
                embed_dim=embedding_dim,
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                qkv_bias=True
            )

            if pretrained_vit_path is not None:
                print(f"Loading pre-trained ViT weights from {pretrained_vit_path}...")
                state_dict = torch.load(pretrained_vit_path, map_location='cpu')
                if 'head.weight' in state_dict and isinstance(self.vit.head, nn.Identity):
                     del state_dict['head.weight']
                     del state_dict['head.bias']
                self.vit.load_state_dict(state_dict, strict=False)

            self.bert = BERTTextEncoder(
                vocab_size=30522,
                hidden_size=512,
                num_layers=6,
                num_heads=8,
                max_seq_length=512,
                dropout=0.1
            )

            if pretrained_bert_path is not None:
                print(f"Loading pre-trained BERT weights from {pretrained_bert_path}...")
                state_dict = torch.load(pretrained_bert_path, map_location='cpu')
                self.bert.load_state_dict(state_dict, strict=False)
            
            self.vit_hidden_size = embedding_dim
            self.bert_hidden_size = self.bert.hidden_size
        
        if freeze_encoders:
            print("Freezing encoder weights...")
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # ViT-Base has 196 patches + 1 CLS token = 197 tokens
        # BERT max length is usually 128 or 512. In train.py it is set to 128.
        # Total sequence length = 197 + 128 = 325.
        # We set max_seq_len to 512 to be safe.
        
        if self.use_fusion:
            fusion_dim = 512
            self.fusion_feature_fusion = MultimodalFeatureFusion(
                bert_dim=self.bert_hidden_size,
                vit_dim=self.vit_hidden_size,
                fusion_dim=fusion_dim,
                depth=2,
                num_heads=8,
                mlp_ratio=4.0,
                dropout=dropout,
                max_seq_len=512
            )
            fusion_output_dim = fusion_dim
        else:
            fusion_output_dim = self.bert_hidden_size + self.vit_hidden_size
        
        # Classification Head
        self.classifier = ClassificationHead(fusion_output_dim, num_classes, dropout)

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass of the multi-modal model.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, 3, 224, 224]
            input_ids (torch.Tensor): BERT input ids [batch_size, seq_len]
            attention_mask (torch.Tensor): BERT attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        if self.use_huggingface:
            # HuggingFace ViT
            vit_outputs = self.vit(pixel_values=images)
            vit_features = vit_outputs.last_hidden_state # [Batch, 197, 768]
            
            # HuggingFace BERT
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_features = bert_outputs.last_hidden_state # [Batch, SeqLen, 768]
        else:
            # Custom ViT forward pass
            # Our custom ViT returns (cls_rep, patch_sequence, logits)
            # We want the patch sequence for fusion
            _, vit_features, _ = self.vit(images, return_patch_sequence=True)
            
            # Custom BERT forward pass
            # BERTTextEncoder returns (sentence_repr, token_seq)
            _, bert_features = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Feature Fusion
        if self.use_fusion:
            # Output Shape: [Batch, TotalSeqLen, FusionDim]
            fused_features = self.fusion_feature_fusion(bert_features, vit_features)
            
            # Pooling Strategy: Mean pooling over the sequence dimension
            pooled_features = fused_features.mean(dim=1) # [Batch, FusionDim]
        else:
            # Pooling Strategy: Use CLS token (index 0) for both
            bert_pooled = bert_features[:, 0, :] # [Batch, 768]
            vit_pooled = vit_features[:, 0, :]   # [Batch, 768]
            
            # Concatenate features
            pooled_features = torch.cat([bert_pooled, vit_pooled], dim=1) # [Batch, 1536]
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
