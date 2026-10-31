import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from baseline_resnet50 import ResNet50Baseline
from rsna_dataset import RSNABoneAgeDataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = BASE_DIR / "metadata" / "train_metadata.csv"
VAL_CSV = BASE_DIR / "metadata" / "validation_metadata.csv"

DATA_ROOT = Path(
    os.environ.get("RSNA_DATA_ROOT", r"D:\Pediatric_Bone_Age\preprocessing")
)

OUTPUT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 0.0001

# Smoke test: use a small subset to verify the complete pipeline.
# Set to None later to train on the full dataset.
TRAIN_LIMIT = 512
VAL_LIMIT = 128

IMAGE_SIZE = 224
NUM_WORKERS = 0
SEED = 42


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# DATA
# --------------------------------------------------

def create_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = RSNABoneAgeDataset(
        metadata_csv=TRAIN_CSV,
        transform=transform,
        data_root=DATA_ROOT
    )

    val_dataset = RSNABoneAgeDataset(
        metadata_csv=VAL_CSV,
        transform=transform,
        data_root=DATA_ROOT
    )

    if TRAIN_LIMIT is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(TRAIN_LIMIT, len(train_dataset)))
        )

    if VAL_LIMIT is not None:
        val_dataset = Subset(
            val_dataset,
            range(min(VAL_LIMIT, len(val_dataset)))
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device, dtype=torch.float32).view(-1)

        optimizer.zero_grad()

        predictions = model(images).view(-1)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.2f}")

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_error = 0.0

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)

        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32).view(-1)

            predictions = model(images).view(-1)
            loss = criterion(predictions, targets)

            total_error += loss.item() * images.size(0)
            progress.set_postfix(mae=f"{loss.item():.2f}")

    return total_error / len(loader.dataset)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_training_pipeline():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("PEDIATRIC BONE AGE - RESNET50 BASELINE")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training limit: {TRAIN_LIMIT}")
    print(f"Validation limit: {VAL_LIMIT}")

    if device.type == "cpu":
        print("\nCPU mode: running a small smoke test.")
    else:
        print("\nGPU detected: CUDA training enabled.")

    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders()

    print("\nInitializing ResNet-50...")
    model = ResNet50Baseline(pretrained=True).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_mae = float("inf")

    print("\nStarting training...")
    print("-" * 55)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_mae = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Training MAE:   {train_loss:.2f} months")
        print(f"Validation MAE: {val_mae:.2f} months")

        if val_mae < best_val_mae:
            best_val_mae = val_mae

            checkpoint_path = OUTPUT_DIR / "best_resnet50.pth"

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": val_mae,
                "image_size": IMAGE_SIZE
            }, checkpoint_path)

            print(f"Best checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 55)
    print("TRAINING FINISHED")
    print(f"Best validation MAE: {best_val_mae:.2f} months")
    print("=" * 55)

    if TRAIN_LIMIT is not None:
        print(
            "\nNOTE: This was a smoke test using a subset. "
            "Do not report this MAE as the final full-dataset result."
        )


if __name__ == "__main__":
    run_training_pipeline()