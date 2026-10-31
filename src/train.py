import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """
    Executes one training epoch with support for dictionary-based DataLoaders.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Handle dictionary batch from Pavan's rsna_dataset.py vs tuple fallback
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            targets = batch["boneage"].to(device, dtype=torch.float32)
        else:
            images, targets = batch
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Evaluates model validation performance (MAE in months) with dictionary DataLoader support.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                targets = batch["boneage"].to(device, dtype=torch.float32)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    device: torch.device = None
):
    """
    Complete training loop: manages optimizer, epoch loop, validation MAE logging, and checkpointing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.L1Loss()  # Mean Absolute Error in months
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "resnet50_baseline_best.pth")

    best_val_mae = float("inf")

    print(f"Starting ResNet-50 Baseline Training on device: {device}")
    print(f"Target Baseline MAE: < 12.0 months\n" + "-" * 50)

    for epoch in range(epochs):
        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mae = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train MAE: {train_mae:.2f} mos | Val MAE: {val_mae:.2f} mos")

        # Save best model checkpoint when validation MAE improves
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
            }, best_checkpoint_path)
            print(f"   --> Saved new best model checkpoint (Val MAE: {val_mae:.2f} months)")

    print("-" * 50)
    print(f"Training Complete. Best Validation MAE: {best_val_mae:.2f} months")
    print(f"Checkpoint saved to: {best_checkpoint_path}")
    return best_val_mae
