import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from src.dataset_t4sa import T4SADataset
from src.encoders.mae import MaskedAutoencoderViT
from src.encoders.vision_transformer import VisionTransformer


def train_one_epoch(model, loader, optimizer, scheduler, device, mask_ratio=0.75):
    model.train()
    running_loss = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad()

        # Forward pass
        loss, _, _ = model(images, mask_ratio=mask_ratio)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def validate(model, loader, device, mask_ratio=0.75):
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            loss, _, _ = model(images, mask_ratio=mask_ratio)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def plot_metrics(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.title("MAE Reconstruction Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Pre-train MAE on T4SA")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=1.5e-4, help="Learning rate (base lr)"
    )
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Masking ratio")
    parser.add_argument(
        "--warmup-epochs", type=int, default=10, help="Number of warmup epochs"
    )

    # ViT-Small configuration (better for 400k samples than Base)
    parser.add_argument(
        "--embed-dim", type=int, default=384, help="Embedding dimension"
    )
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument(
        "--num-heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument(
        "--decoder-embed-dim", type=int, default=256, help="Decoder embedding dimension"
    )
    parser.add_argument("--decoder-depth", type=int, default=8, help="Decoder depth")
    parser.add_argument(
        "--decoder-num-heads", type=int, default=8, help="Decoder number of heads"
    )

    args = parser.parse_args()

    # Define Transforms
    # MAE needs strong augmentation: RandomResizedCrop
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Setup Data
    # We load dataset twice to apply different transforms to train and val
    # This is slightly inefficient in loading but ensures correct transforms
    print("Loading datasets...")
    try:
        # Load full dataset with train transform
        full_train_dataset = T4SADataset(transform=train_transform)
        # Load full dataset with val transform
        full_val_dataset = T4SADataset(transform=val_transform)

        print(f"Loaded T4SA dataset with {len(full_train_dataset)} samples.")
    except Exception as e:
        print(f"Error loading T4SA dataset: {e}")
        return

    # Split indices
    dataset_size = len(full_train_dataset)
    # Generate random permutation of indices
    indices = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(42)
    ).tolist()
    train_size = int(0.9 * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create Subsets with correct transforms
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Vision Transformer (ViT-Small by default)
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        num_classes=0,
    )

    # Initialize MAE Model
    model = MaskedAutoencoderViT(
        vit=vit,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=True,
    )
    model.to(device)

    # Effective LR = base_lr * batch_size / 256
    eff_lr = args.lr * args.batch_size / 256
    optimizer = optim.AdamW(
        model.parameters(), lr=eff_lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    # Scheduler with Warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Setup output directory
    output_dir = os.path.join("pretrained_encoders", "mae_small")
    os.makedirs(output_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Starting MAE pre-training (ViT-Small) for {args.epochs} epochs...")
    print(
        f"Config: Embed Dim={args.embed_dim}, Heads={args.num_heads}, Depth={args.depth}"
    )

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            mask_ratio=args.mask_ratio,
        )
        val_loss = validate(model, val_loader, device, mask_ratio=args.mask_ratio)

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
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_mae_model.pth")
            )
            torch.save(
                model.vit.state_dict(), os.path.join(output_dir, "best_vit_encoder.pth")
            )
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"mae_checkpoint_epoch_{epoch + 1}.pth"),
            )

    # Save final metrics plot
    plot_path = os.path.join(output_dir, "mae_training_metrics.png")
    plot_metrics(train_losses, val_losses, plot_path)
    print(f"Training complete. Metrics saved to {plot_path}")


if __name__ == "__main__":
    main()
