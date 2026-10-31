import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from rsna_dataset import RSNABoneAgeDataset
    from baseline_resnet50 import ResNet50Baseline
except ModuleNotFoundError:
    from src.rsna_dataset import RSNABoneAgeDataset
    from src.baseline_resnet50 import ResNet50Baseline


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_METADATA = BASE_DIR / "metadata" / "train_metadata.csv"
VAL_METADATA = BASE_DIR / "metadata" / "validation_metadata.csv"

# Check local Windows path vs Colab/Data fallback
default_local_path = r"D:\Pediatric_Bone_Age\preprocessing"
fallback_colab_path = BASE_DIR / "data"

if os.path.exists(default_local_path):
    default_data_root = default_local_path
else:
    default_data_root = str(fallback_colab_path)

DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", default_data_root))

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42


# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================================
# 3. DEVICE & SETUP
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("PEDIATRIC BONE AGE - RESNET50 BASELINE")
print("=" * 55)
print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ==========================================================
# 4. IMAGE TRANSFORMATIONS
# ==========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 5. TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:

    model.train()
    running_loss = 0.0

    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            targets = batch["boneage"].to(device, dtype=torch.float32)
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images).reshape(-1)
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


# ==========================================================
# 6. VALIDATION / EVALUATION
# ==========================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):

    model.eval()
    running_loss = 0.0
    total_absolute_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                targets = batch["boneage"].to(device, dtype=torch.float32)
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device, dtype=torch.float32)

            outputs = model(images).reshape(-1)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            total_absolute_error += torch.abs(outputs - targets).sum().item()

    validation_loss = running_loss / len(dataloader.dataset)
    validation_mae = total_absolute_error / len(dataloader.dataset)

    return validation_loss, validation_mae


# ==========================================================
# 7. COMPLETE TRAINING PIPELINE
# ==========================================================

def run_training_pipeline():

    if not TRAIN_METADATA.is_file() or not VAL_METADATA.is_file():
        raise FileNotFoundError(f"Metadata CSV files missing in {BASE_DIR / 'metadata'}")

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Processed image directory not found: {DATA_ROOT}")

    print("\nLoading datasets...")
    train_dataset = RSNABoneAgeDataset(metadata_csv=TRAIN_METADATA, transform=transform, data_root=DATA_ROOT)
    val_dataset = RSNABoneAgeDataset(metadata_csv=VAL_METADATA, transform=transform, data_root=DATA_ROOT)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    print("\nInitializing ResNet-50 model...")
    model = ResNet50Baseline(pretrained=True).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_mae = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "resnet50_baseline_best.pth"

    print("\nStarting training...")
    print("Target baseline MAE: below 12.0 months")
    print("-" * 55)

    for epoch in range(EPOCHS):
        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch [{epoch + 1:02d}/{EPOCHS:02d}]")
        print(f"Training MAE: {train_mae:.2f} months")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation MAE: {val_mae:.2f} months")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": val_mae,
                "validation_loss": val_loss
            }, checkpoint_path)
            print("--> Best model checkpoint saved!")

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)
    print(f"Best Validation MAE: {best_val_mae:.2f} months")
    print("Checkpoint:", checkpoint_path)

    return best_val_mae


if __name__ == "__main__":
    run_training_pipeline()
