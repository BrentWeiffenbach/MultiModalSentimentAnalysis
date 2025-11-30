import torch
import torch.nn as nn

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
    