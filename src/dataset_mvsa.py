import os
import torch
import matplotlib.pyplot as plt
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
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Removed to match T4SA training
        ])
        
        # Read the labels file
        self.samples = []
        invalid_count = 0
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:] 
            for line in lines:
                parts = line.strip().split('\t')
                # MVSA-Single has fewer columns, but ID is always first
                if len(parts) < 2: 
                    continue
                sample_id = parts[0].strip()
                
                # Extract sentiments (text and image)
                # MVSA-Single: ID, sentiment (column 1)
                sentiments = []
                
                # MVSA-Single format: ID \t sentiment
                # sentiment is like "positive,neutral" (text, image)
                if len(parts) > 1 and ',' in parts[1]:
                    pair = parts[1].strip().split(',')
                    if len(pair) >= 2:
                        sentiments.append(pair[0].strip()) # Text sentiment
                        sentiments.append(pair[1].strip()) # Image sentiment
                
                if len(sentiments) == 0:
                    continue
                
                img_path = os.path.join(root_dir, f"{sample_id}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(root_dir, f"{sample_id}.png")

                text_path = os.path.join(root_dir, f"{sample_id}.txt")
                if not (os.path.exists(img_path) and os.path.exists(text_path)):
                    continue
                
                # Validate image can be opened
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's a valid image
                    # Re-open after verify (verify closes the file)
                    with Image.open(img_path) as img:
                        img.convert("RGB")  # Test conversion
                    self.samples.append((img_path, text_path, sentiments))
                except Exception as e:
                    print(f"Skipping invalid image {sample_id}: {e}")
                    invalid_count += 1
                    continue
        
        print(f"Loaded {len(self.samples)} valid samples from {root_dir}, skipped {invalid_count} invalid samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            img_path, text_path, sentiments = self.samples[idx]

            # Load image
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            # Load text
            # Try utf-8 first, then latin-1 if that fails
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(text_path, 'r', encoding='latin-1') as f:
                    text = f.read().strip()

            # Calculate percentage-based label from all annotators
            # Count occurrences of each sentiment
            positive_count = sentiments.count("positive")
            neutral_count = sentiments.count("neutral")
            negative_count = sentiments.count("negative")
            total = len(sentiments)
            
            # Create label as percentages [negative, neutral, positive] to match T4SA order (NEG, NEU, POS)
            soft_label = torch.tensor([
                negative_count / total,
                neutral_count / total,
                positive_count / total
            ], dtype=torch.float)
            
            # Determine dominant class index
            class_idx = torch.argmax(soft_label).item()
            
            # Helper to map string to index
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            # We assume sentiments list has [text_sentiment, image_sentiment] order from the parsing logic
            # The parsing logic was: sentiments.append(pair[0]) (Text), sentiments.append(pair[1]) (Image)
            text_label_idx = label_map.get(sentiments[0], -1)
            image_label_idx = label_map.get(sentiments[1], -1)

            return {
                "image": image, 
                "text": text, 
                "label": class_idx, 
                "soft_label": soft_label,
                "text_label": text_label_idx,
                "image_label": image_label_idx
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Try next sample
            idx = (idx + 1) % len(self.samples)
            return self.__getitem__(idx)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    # Always load Single
    single_dir = os.path.join(root_dir, "MVSA_Single")
    data_dir = os.path.join(single_dir, "data")
    labels_file = os.path.join(single_dir, "labelResultAll.txt")
    
    if os.path.exists(data_dir) and os.path.exists(labels_file):
        print(f"Loading MVSA-Single from {data_dir}...")
        dataset = MVSADataset(data_dir, labels_file)
    else:
        print(f"Error: MVSA-Single not found at {single_dir}")
        exit(1)

    print(f"Dataset size: {len(dataset)}")

    # 1. Analyze Label Distribution (Fast method using cached metadata)
    print("Analyzing label distribution...")
    label_counts = {0: 0, 1: 0, 2: 0} # 0: Positive, 1: Neutral, 2: Negative
    
    all_samples = dataset.samples

    for _, _, sentiments in all_samples:
        positive_count = sentiments.count("positive")
        neutral_count = sentiments.count("neutral")
        negative_count = sentiments.count("negative")
        total = len(sentiments)
        
        if total == 0:
            continue
        
        # Calculate soft label
        probs = [positive_count / total, neutral_count / total, negative_count / total]
        # Determine dominant class
        dominant_class = probs.index(max(probs))
        label_counts[dominant_class] += 1

    classes = ['Positive', 'Neutral', 'Negative']
    counts = [label_counts[0], label_counts[1], label_counts[2]]
    total_samples = sum(counts)
    percentages = [c / total_samples * 100 for c in counts]

    print("\nClass Distribution (Dominant Label):")
    for cls, count, pct in zip(classes, counts, percentages):
        print(f"{cls}: {count} ({pct:.2f}%)")

    # Plot distribution
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['green', 'gray', 'red'])
    plt.title('Sentiment Class Distribution (MVSA-Single)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\\n({height/total_samples*100:.1f}%)',
                ha='center', va='bottom')
                
    plt.savefig('label_distribution_single.png')
    print("Saved label_distribution_single.png")

    # 2. Visualize Random Samples
    print("\nVisualizing random samples...")
    # Use a small batch size to get a few samples
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    images = batch['image']
    texts = batch['text']
    soft_labels = batch['soft_label']  # Use soft_label for probabilities
    labels = batch['label']            # Use label for dominant class index

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        # Use soft_label for probabilities
        label_probs = soft_labels[i]
        label_str = f"Pos: {label_probs[2]:.2f}, Neu: {label_probs[1]:.2f}, Neg: {label_probs[0]:.2f}"
        dominant_idx = labels[i].item() if hasattr(labels[i], 'item') else int(labels[i])
        dominant = classes[dominant_idx]

        ax.imshow(img)
        ax.set_title(f"Label: {dominant}\n{label_str}", fontsize=10)
        ax.axis('off')

        # Wrap text for display
        text_content = texts[i]
        wrapped_text = ""
        words = text_content.split()
        line = ""
        for word in words:
            if len(line) + len(word) > 40:
                wrapped_text += line + "\n"
                line = word + " "
            else:
                line += word + " "
        wrapped_text += line

        ax.text(0.5, -0.05, wrapped_text, ha='center', va='top', transform=ax.transAxes, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.savefig('sample_visualization_single.png')
    print("Saved sample_visualization_single.png")