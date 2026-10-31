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


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = BASE_DIR / "metadata" / "train_metadata.csv"
VAL_CSV = BASE_DIR / "metadata" / "validation_metadata.csv"

DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        r"D:\Pediatric_Bone_Age\preprocessing"
    )
)

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Smoke-test settings for CPU
BATCH_SIZE = 4
EPOCHS = 2
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

# Use small subsets initially.
# Set both to None for full-dataset training.
TRAIN_LIMIT = 512
VAL_LIMIT = 128

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATASET AND DATALOADERS
# =========================================================

def create_dataloaders():

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training metadata not found: {TRAIN_CSV}")

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation metadata not found: {VAL_CSV}")

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

    # Subset only for smoke testing
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


# =========================================================
# EXTRACT DICTIONARY BATCH
# =========================================================

def get_batch(batch, device):

    # Your RSNABoneAgeDataset returns a dictionary.
    images = batch["image"].to(device)

    targets = batch["boneage"].to(
        device,
        dtype=torch.float32
    ).view(-1)

    return images, targets


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_absolute_error = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Training")

    for batch in progress:

        images, targets = get_batch(batch, device)

        optimizer.zero_grad()

        predictions = model(images).view(-1)

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_absolute_error += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.2f}"
        )

    return total_absolute_error / total_samples


# =========================================================
# VALIDATION
# =========================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation")

        for batch in progress:

            images, targets = get_batch(batch, device)

            predictions = model(images).view(-1)

            loss = criterion(predictions, targets)

            batch_size = images.size(0)

            total_absolute_error += loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix(
                mae=f"{loss.item():.2f}"
            )

    return total_absolute_error / total_samples


# =========================================================
# MAIN TRAINING PIPELINE
# =========================================================

def run_training_pipeline():

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("PEDIATRIC BONE AGE - RESNET50 BASELINE")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    if device.type == "cpu":
        print("\nCPU detected: running smoke test.")
    else:
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")

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
    print("=" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_mae = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\nTraining MAE:   {train_mae:.2f} months")
        print(f"Validation MAE: {val_mae:.2f} months")

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            checkpoint_path = (
                CHECKPOINT_DIR / "best_resnet50.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_mae": val_mae,
                    "image_size": IMAGE_SIZE
                },
                checkpoint_path
            )

            print(f"Saved best model: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Best validation MAE: {best_val_mae:.2f} months")
    print("=" * 60)

    if TRAIN_LIMIT is not None:
        print(
            "\nNOTE: Smoke-test result only. "
            "This is not the final full-dataset baseline."
        )


if __name__ == "__main__":
    run_training_pipeline()