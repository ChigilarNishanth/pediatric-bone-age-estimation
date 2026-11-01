import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Enable CuDNN auto-tuner for maximum GPU speed on fixed image dimensions
torch.backends.cudnn.benchmark = True

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from rsna_dataset import RSNABoneAgeDataset
    from baseline_resnet50 import ResNet50Baseline
except ModuleNotFoundError:
    from src.rsna_dataset import RSNABoneAgeDataset
    from src.baseline_resnet50 import ResNet50Baseline

TRAIN_METADATA = BASE_DIR / "metadata" / "train_metadata.csv"
VAL_METADATA = BASE_DIR / "metadata" / "validation_metadata.csv"

DEFAULT_PREPROCESSING_DIR = BASE_DIR / "preprocessing"
DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", DEFAULT_PREPROCESSING_DIR))

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# OPTIMIZED FP32 GPU CONFIGURATION
BATCH_SIZE = 64         # Doubled for full T4 GPU utilization
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["boneage"].to(device, dtype=torch.float32, non_blocking=True)
        else:
            images, targets = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images).reshape(-1)
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Print progress every 50 batches for rapid feedback (~15s intervals)
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            current_mae = loss.item()
            print(f"  [Epoch {epoch:02d}/{EPOCHS:02d}] Batch {batch_idx + 1:03d}/{total_batches:03d} | Batch Loss: {current_mae:.2f} mos")

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_absolute_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["boneage"].to(device, dtype=torch.float32, non_blocking=True)
            else:
                images, targets = batch
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, dtype=torch.float32, non_blocking=True)

            outputs = model(images).reshape(-1)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            total_absolute_error += torch.abs(outputs - targets).sum().item()

    validation_loss = running_loss / len(dataloader.dataset)
    validation_mae = total_absolute_error / len(dataloader.dataset)

    return validation_loss, validation_mae

def run_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(" PEDIATRIC BONE AGE - RESNET-50 FULL TRAINING (REVIEW 2)")
    print("=" * 65)
    print(f"Active Hardware      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Precision Mode       : Full 32-Bit Floating Point (FP32)")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Epochs               : {EPOCHS}")
    print(f"Learning Rate        : {LEARNING_RATE}")
    print(f"Metadata Path        : {TRAIN_METADATA}")
    print(f"Preprocessing Root   : {DATA_ROOT}")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("\n [WARNING] CUDA IS NOT ACTIVE! Change Colab runtime to T4 GPU before training.")

    if not TRAIN_METADATA.is_file() or not VAL_METADATA.is_file():
        raise FileNotFoundError(f"Metadata CSVs missing in {BASE_DIR / 'metadata'}")

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Preprocessed image directory not found at '{DATA_ROOT}'")

    train_dataset = RSNABoneAgeDataset(metadata_csv=TRAIN_METADATA, transform=transform, data_root=DATA_ROOT)
    val_dataset = RSNABoneAgeDataset(metadata_csv=VAL_METADATA, transform=transform, data_root=DATA_ROOT)

    print(f"\nTrain samples       : {len(train_dataset)}")
    print(f"Validation samples  : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if (NUM_WORKERS > 0 and torch.cuda.is_available()) else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if (NUM_WORKERS > 0 and torch.cuda.is_available()) else False
    )

    model = ResNet50Baseline(pretrained=True).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_mae = float("inf")
    best_checkpoint_path = CHECKPOINT_DIR / "resnet50_baseline_best.pth"

    print("\nStarting Full Training Execution...")
    print("Target Metric: Validation MAE < 12.0 Months")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n>>> Epoch [{epoch:02d}/{EPOCHS:02d}] Execution Started")
        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        print(f"--> Epoch [{epoch:02d}/{EPOCHS:02d}] Complete | Train MAE: {train_mae:.2f} mos | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} mos")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": val_mae,
                "validation_loss": val_loss
            }, best_checkpoint_path)
            print(f"    ⭐ Best model checkpoint updated -> {best_checkpoint_path.name} (Val MAE: {val_mae:.2f} mos)")

    print("-" * 65)
    print("FULL TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best Validation MAE Achieved : {best_val_mae:.2f} months")
    print(f"Saved Checkpoint Path       : {best_checkpoint_path}")
    print("=" * 65)

    return best_val_mae

if __name__ == "__main__":
    run_training_pipeline()
