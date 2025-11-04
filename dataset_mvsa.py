import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

class MVSADataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Read the labels file
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) < 2: 
                    continue
                sample_id = parts[0].strip()
                label_str = parts[1].split(',')[0].strip()  # e.g. "positive"
                
                img_path = os.path.join(root_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(root_dir, f"{sample_id}.png")

                text_path = os.path.join(root_dir, f"{sample_id}.txt")
                if not (os.path.exists(img_path) and os.path.exists(text_path)):
                    continue
                
                self.samples.append((img_path, text_path, label_str))
        
        self.label_map = {
            "positive": torch.tensor([1, 0, 0], dtype=torch.float),
            "neutral":  torch.tensor([0, 1, 0], dtype=torch.float),
            "negative": torch.tensor([0, 0, 1], dtype=torch.float),
        }

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text_path, label_str = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        label = self.label_map.get(label_str, torch.tensor([0, 0, 0], dtype=torch.float))

        return {"image": image, "text": text, "label": label}


if __name__ == "__main__":
    dataset = MVSADataset("/home/rweiffenbach/MVSA/data", "/home/rweiffenbach/MVSA/labelResultAll.txt")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        print(batch["image"].shape)   # e.g. [4, 3, 224, 224]
        print(batch["text"])          # list of 4 captions
        print(batch["label"])         # [4, 3]
        break
