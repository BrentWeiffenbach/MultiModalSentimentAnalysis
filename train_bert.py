import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, get_cosine_schedule_with_warmup

from src.dataset_t4sa import T4SADataset
from src.encoders.bert_encoder import BERTTextEncoder
from src.encoders.bert_mlm import BertForMaskedLM


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train_one_epoch(model, loader, optimizer, scheduler, device, tokenizer):
    model.train()
    running_loss = 0.0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Create masked inputs and labels
        input_ids, labels = mask_tokens(input_ids, tokenizer)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def validate(model, loader, device, tokenizer):
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            input_ids, labels = mask_tokens(input_ids, tokenizer)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

            running_loss += loss.item() * input_ids.size(0)
            total += input_ids.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def plot_metrics(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.title("BERT MLM Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Pre-train BERT on T4SA")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Hidden size of BERT"
    )
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Number of warmup epochs"
    )
    args = parser.parse_args()

    # Setup Data
    try:
        full_dataset = T4SADataset()
        print(f"Loaded T4SA dataset with {len(full_dataset)} samples.")
    except Exception as e:
        print(f"Error loading T4SA dataset: {e}")
        return

    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize BERT Encoder
    # Using Medium-Small model configuration (512/6/8) ~30M params
    bert_encoder = BERTTextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=512,
        dropout=args.dropout,
    )

    # Initialize MLM Model
    model = BertForMaskedLM(bert_encoder)
    model.to(device)

    # Add weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler with Warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Setup output directory
    output_dir = os.path.join("pretrained_encoders", "bert_medium")
    os.makedirs(output_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Starting BERT pre-training for {args.epochs} epochs...")
    print(
        f"Config: Hidden={args.hidden_size}, Layers={args.num_layers}, Heads={args.num_heads}"
    )

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, tokenizer
        )
        val_loss = validate(model, val_loader, device, tokenizer)

        end_time = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {end_time - start_time:.2f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save full MLM model
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_bert_mlm_model.pth")
            )
            # Save just the BERT encoder weights
            torch.save(
                model.bert.state_dict(),
                os.path.join(output_dir, "best_bert_encoder.pth"),
            )
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.bert.state_dict(),
                os.path.join(
                    output_dir, f"bert_encoder_checkpoint_epoch_{epoch + 1}.pth"
                ),
            )

    # Save final metrics plot
    plot_path = os.path.join(output_dir, "bert_training_metrics.png")
    plot_metrics(train_losses, val_losses, plot_path)
    print(f"Training complete. Metrics saved to {plot_path}")


if __name__ == "__main__":
    main()
