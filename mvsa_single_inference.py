import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from tqdm import tqdm
import random

from src.multimodal_sentiment_analysis import MultiModalSentimentAnalysis
from src.dataset_mvsa import MVSADataset
from transformers import BertTokenizer

def visualize_batch(samples, output_path):
    # samples is a list of dicts with keys: image, text, prediction, probabilities, true_label
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    classes = ["Negative", "Neutral", "Positive"]
    
    for i, sample in enumerate(samples):
        if i >= 4: break
        
        ax = axes[i]
        
        # Image
        img = sample['image'].permute(1, 2, 0).cpu().numpy()
        # Clip to 0-1 just in case
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Text info
        pred_class = classes[sample['prediction']]
        true_class = classes[sample['true_label']]
        probs = sample['probabilities']
        
        text = sample['text']
        wrapped_text = textwrap.fill(text, width=40)
        
        # Color code title based on correctness
        title_color = 'green' if sample['prediction'] == sample['true_label'] else 'red'
        
        info_text = (
            f"True: {true_class} | Pred: {pred_class}\n"
            f"Conf: Neg: {probs[0]:.2f}, Neu: {probs[1]:.2f}, Pos: {probs[2]:.2f}\n\n"
            f"{wrapped_text}"
        )
        
        ax.set_title(info_text, fontsize=12, color=title_color, loc='left', wrap=True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Inference on MVSA-Single Dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--mvsa-path", type=str, default="MVSA_Single", help="Path to MVSA_Single folder")
    parser.add_argument("--output-path", type=str, default="mvsa_results.png", help="Path to save visualization")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Setup Dataset
    # We need to ensure MVSADataset matches T4SA transforms (No Normalize)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    data_dir = os.path.join(args.mvsa_path, "data")
    labels_file = os.path.join(args.mvsa_path, "labelResultAll.txt")
    
    print(f"Loading MVSA dataset from {data_dir}...")
    dataset = MVSADataset(data_dir, labels_file, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Loaded MVSA dataset with {len(dataset)} samples")
    
    # Initialize Model
    # Assuming custom model based on previous context (ViT-Small + BERT-Medium + Fusion)
    model = MultiModalSentimentAnalysis(
        num_classes=3,
        dropout=0.1,
        freeze_encoders=False,
        use_huggingface=False, 
        use_fusion=True
    )
    
    # Load Weights
    print(f"Loading weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # Metrics for consistent samples (Text == Image)
    correct_consistent = 0
    total_consistent = 0
    
    # Metrics for inconsistent samples
    correct_inconsistent = 0
    total_inconsistent = 0
    
    # Metrics against specific modalities
    correct_text = 0
    correct_image = 0
    
    samples_for_viz = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device) # Integer labels
            soft_labels = batch['soft_label'] # [Batch, 3]
            text_labels = batch['text_label'].to(device)
            image_labels = batch['image_label'].to(device)
            
            # Tokenize text
            encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Check consistency based on soft_label max value
            # If max == 1.0, it means both annotators agreed (or 100% of them)
            # If max < 1.0 (e.g. 0.5), there was disagreement
            max_soft, _ = torch.max(soft_labels, dim=1)
            is_consistent = (max_soft > 0.99)
            
            for i in range(len(labels)):
                pred = predictions[i].item()
                label = labels[i].item()
                consistent = is_consistent[i].item()
                
                if pred == label:
                    correct += 1
                    if consistent:
                        correct_consistent += 1
                    else:
                        correct_inconsistent += 1
                
                if pred == text_labels[i].item():
                    correct_text += 1
                
                if pred == image_labels[i].item():
                    correct_image += 1
                
                total += 1
                if consistent:
                    total_consistent += 1
                else:
                    total_inconsistent += 1

            # Collect random samples (just take first 4 from first batch, or random ones)
            if len(samples_for_viz) < 4:
                for i in range(len(labels)):
                    if len(samples_for_viz) >= 4: break
                    samples_for_viz.append({
                        'image': images[i].cpu(),
                        'text': texts[i],
                        'prediction': predictions[i].item(),
                        'true_label': labels[i].item(),
                        'probabilities': probs[i].cpu().numpy()
                    })
    
    accuracy = correct / total
    acc_consistent = correct_consistent / total_consistent if total_consistent > 0 else 0
    acc_inconsistent = correct_inconsistent / total_inconsistent if total_inconsistent > 0 else 0
    acc_text = correct_text / total
    acc_image = correct_image / total
    
    print(f"MVSA-Single Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  - Consistent Samples (Text==Image): {acc_consistent:.4f} ({correct_consistent}/{total_consistent})")
    print(f"  - Inconsistent Samples (Text!=Image): {acc_inconsistent:.4f} ({correct_inconsistent}/{total_inconsistent})")
    print(f"  - Accuracy vs Text Label: {acc_text:.4f} ({correct_text}/{total})")
    print(f"  - Accuracy vs Image Label: {acc_image:.4f} ({correct_image}/{total})")
    
    visualize_batch(samples_for_viz, args.output_path)

if __name__ == "__main__":
    main()
