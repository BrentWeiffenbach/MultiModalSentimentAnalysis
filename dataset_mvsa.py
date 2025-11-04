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
            lines = f.readlines()[1:] 
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) < 4: 
                    continue
                sample_id = parts[0].strip()
                
                # Extract text sentiment from all 3 annotators
                text_sentiments = []
                for i in range(1, 4):  # columns 1, 2, 3 (the 3 annotators)
                    if ',' in parts[i]:
                        text_sentiment = parts[i].split(',')[0].strip()  # Get text part
                        text_sentiments.append(text_sentiment)
                
                if len(text_sentiments) == 0:
                    continue
                
                img_path = os.path.join(root_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(root_dir, f"{sample_id}.png")

                text_path = os.path.join(root_dir, f"{sample_id}.txt")
                if not (os.path.exists(img_path) and os.path.exists(text_path)):
                    continue
                
                self.samples.append((img_path, text_path, text_sentiments))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text_path, text_sentiments = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Calculate percentage-based label from all annotators
        # Count occurrences of each sentiment
        positive_count = text_sentiments.count("positive")
        neutral_count = text_sentiments.count("neutral")
        negative_count = text_sentiments.count("negative")
        total = len(text_sentiments)
        
        # Create label as percentages [positive, neutral, negative]
        label = torch.tensor([
            positive_count / total,
            neutral_count / total,
            negative_count / total
        ], dtype=torch.float)

        return {"image": image, "text": text, "label": label}


if __name__ == "__main__":
    dataset = MVSADataset("/home/rweiffenbach/MVSA/data", "/home/rweiffenbach/MVSA/labelResultAll.txt")
    # dataset = MVSADataset("/home/brent/cs541/MultiModalSentimentAnalysis/MVSA/data", "/home/brent/cs541/MultiModalSentimentAnalysis/MVSA/labelResultAll.txt")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        print(batch["image"].shape)   # e.g. [4, 3, 224, 224]
        print(batch["text"])          # list of 4 captions
        print(batch["label"])         # [4, 3]
        break
