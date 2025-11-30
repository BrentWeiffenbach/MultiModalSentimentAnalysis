import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import time

from src.dataset_t4sa import T4SADataset
from src.multimodal_sentiment_analysis import MultiModalSentimentAnalysis

# Hyperparameters
# BATCH_SIZE is now set via command line argument, default is 32 to avoid OOM
LEARNING_RATE = 5e-5
DROPOUT = 0.1
NUM_EPOCHS = 50
PATIENCE = 5

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # Convert soft labels to hard labels for accuracy and CrossEntropy
        labels = batch['labels'].to(device)
        targets = torch.argmax(labels, dim=1)
        
        optimizer.zero_grad()
        
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            targets = torch.argmax(labels, dim=1)
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Sentiment Analysis Training')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--freeze-encoders', action='store_true', help='Freeze ViT and BERT encoders')
    parser.add_argument('--pretrained-vit-path', type=str, default=None, help='Path to pre-trained ViT weights')
    parser.add_argument('--pretrained-bert-path', type=str, default=None, help='Path to pre-trained BERT weights')
    parser.add_argument('--use-huggingface', action='store_true', help='Use HuggingFace pre-trained models')
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    freeze_encoders = args.freeze_encoders
    pretrained_vit_path = args.pretrained_vit_path
    pretrained_bert_path = args.pretrained_bert_path
    use_huggingface = args.use_huggingface
    
    # Setup Data
    # T4SA Dataset assumes data is in T4SA/ folder relative to CWD or specified paths
    # We use defaults from T4SADataset
    
    try:
        full_dataset = T4SADataset()
        print(f"Loaded T4SA dataset with {len(full_dataset)} samples.")
    except Exception as e:
        print(f"Error loading T4SA dataset: {e}")
        return
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        # item['soft_label'] is numpy array, convert to tensor
        labels = torch.stack([torch.tensor(item['soft_label'], dtype=torch.float) for item in batch])
        
        encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        return {
            'images': images,
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # T4SA Class Weights (Inverse Frequency)
    # Negative: 133161 (34.63%) -> W ~ 2.89
    # Neutral: 134435 (34.96%) -> W ~ 2.86
    # Positive: 116931 (30.41%) -> W ~ 3.29
    # Order in T4SA labels is NEG, NEU, POS
    class_weights = torch.tensor([2.89, 2.86, 3.29]).to(device)
    print(f"Using class weights: {class_weights}")
    
    print(f"\nStarting training with: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Dropout={DROPOUT}, Freeze={freeze_encoders}, Fusion=True")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Initialize Model
    model = MultiModalSentimentAnalysis(
        num_classes=3, 
        dropout=DROPOUT, 
        freeze_encoders=freeze_encoders,
        pretrained_vit_path=pretrained_vit_path,
        pretrained_bert_path=pretrained_bert_path,
        use_huggingface=use_huggingface,
        use_fusion=True
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define parameter groups for differential learning rates
    # Higher LR for the classifier head, lower LR for the pre-trained backbones
    # Since we are freezing encoders, we only train the head and fusion module
    # But if freeze_encoders is False, we might want differential LRs.
    # For this specific run, we are freezing encoders, so optimizer will only see requires_grad=True params
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Setup output directory
    output_dir = os.path.join("experiments", "final_training")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        end_time = time.time()
        
        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {end_time-start_time:.2f}s")
        
        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement in validation accuracy for {patience_counter} epochs.")
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
    # Save Results
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plot_metrics(train_losses, val_losses, train_accs, val_accs, plot_path)
    print(f"Saved metrics plot to {plot_path}")

if __name__ == "__main__":
    main()
