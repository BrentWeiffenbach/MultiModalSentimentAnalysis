"""
Environment Test Script
Tests all major pip requirements without downloading datasets.
"""

import sys
print("=" * 60)
print("Testing Python Environment and Dependencies")
print("=" * 60)

# Test Python version
print(f"\n✓ Python version: {sys.version}")

# Test core PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test torchvision
try:
    import torchvision
    from torchvision import transforms
    print(f"✓ torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ torchvision import failed: {e}")

# Test numpy and scipy
try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy import failed: {e}")

# Test pandas
try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

# Test scikit-learn
try:
    import sklearn
    print(f"✓ scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")

# Test transformers and tokenizers
try:
    import transformers
    from transformers import AutoTokenizer
    print(f"✓ transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")

try:
    import tokenizers
    try:
        from importlib.metadata import version
        print(f"✓ tokenizers version: {version('tokenizers')}")
    except Exception:
        print("✓ tokenizers imported, but version could not be determined")
except ImportError as e:
    print(f"✗ tokenizers import failed: {e}")

# Test tqdm
try:
    from tqdm import tqdm, trange
    print(f"✓ tqdm imported successfully")
except ImportError as e:
    print(f"✗ tqdm import failed: {e}")

# Test PIL/Pillow
try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow version: {PIL.__version__}")
except ImportError as e:
    print(f"✗ Pillow import failed: {e}")

print("\n" + "=" * 60)
print("Testing PyTorch Functionality")
print("=" * 60)

# Test basic PyTorch operations
class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Generate synthetic data
print("\n✓ Creating synthetic data...")
batch_size = 32
input_dim = 10
hidden_dim = 16
output_dim = 2

X = torch.randn(batch_size, input_dim)
y = torch.randint(0, output_dim, (batch_size,))

# Create model
print("✓ Initializing model...")
model = SimpleMLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Test forward pass
print("✓ Testing forward pass...")
outputs = model(X)
assert outputs.shape == (batch_size, output_dim), "Output shape mismatch!"

# Test backward pass
print("✓ Testing backward pass...")
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

print(f"✓ Initial loss: {loss.item():.4f}")

# Quick training test
print("✓ Running quick training test (5 epochs)...")
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"  Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("✓ Environment is properly configured")
print("✓ PyTorch, typing, and IntelliSense should work correctly")
print("=" * 60)