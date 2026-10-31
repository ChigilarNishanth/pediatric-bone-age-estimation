
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Support execution both directly and as a package
try:
    from baseline_resnet50 import ResNet50Baseline
    from rsna_dataset import RSNABoneAgeDataset
except ImportError:
    from src.baseline_resnet50 import ResNet50Baseline
    from src.rsna_dataset import RSNABoneAgeDataset


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_METADATA = BASE_DIR / "metadata" / "train_metadata.csv"
VAL_METADATA = BASE_DIR / "metadata" / "validation_metadata.csv"

DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        r"D:\Pediatric_Bone_Age\preprocessing"
    )
)

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet50_baseline_best.pth"

# Smoke-test configuration
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

IMAGE_SIZE = 224

# Small subsets for initial pipeline verification
TRAIN_LIMIT = 512
VAL_LIMIT = 128

RANDOM_SEED = 42


# ==========================================================
# 2. DEVICE CONFIGURATION
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# 3. REPRODUCIBILITY
# ==========================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# 4. IMAGE TRANSFORMATIONS
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 5. DATASET AND DATALOADERS
# ==========================================================

def create_dataloaders():

    if not TRAIN_METADATA.is_file():
        raise FileNotFoundError(
            f"Training metadata not found: {TRAIN_METADATA}"
        )

    if not VAL_METADATA.is_file():
        raise FileNotFoundError(
            f"Validation metadata not found: {VAL_METADATA}"
        )

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Processed image directory not found: {DATA_ROOT}"
        )

    print("\nLoading datasets...")

    train_dataset = RSNABoneAgeDataset(
        metadata_csv=TRAIN_METADATA,
        transform=transform,
        data_root=DATA_ROOT
    )

    val_dataset = RSNABoneAgeDataset(
        metadata_csv=VAL_METADATA,
        transform=transform,
        data_root=DATA_ROOT
    )

    # Select reproducible random subsets for smoke testing
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_indices = torch.randperm(
        len(train_dataset),
        generator=generator
    )[:min(TRAIN_LIMIT, len(train_dataset))].tolist()

    val_indices = torch.randperm(
        len(val_dataset),
        generator=generator
    )[:min(VAL_LIMIT, len(val_dataset))].tolist()

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    return train_loader, val_loader


# ==========================================================
# 6. TRAIN ONE EPOCH
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
    total_samples = 0

    for batch_index, batch in enumerate(dataloader, start=1):

        # Support dictionary-based and tuple-based datasets
        if isinstance(batch, dict):

            images = batch["image"].to(device)

            targets = batch["boneage"].to(
                device,
                dtype=torch.float32
            )

        else:

            images, targets = batch

            images = images.to(device)

            targets = targets.to(
                device,
                dtype=torch.float32
            )

        targets = targets.reshape(-1)

        optimizer.zero_grad()

        outputs = model(images).reshape(-1)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        print(
            f"\rTraining batch "
            f"{batch_index}/{len(dataloader)} "
            f"| Loss: {loss.item():.2f}",
            end="",
            flush=True
        )

    print()

    return running_loss / total_samples


# ==========================================================
# 7. VALIDATION / EVALUATION
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
    total_samples = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader, start=1):

            # Support dictionary-based and tuple-based datasets
            if isinstance(batch, dict):

                images = batch["image"].to(device)

                targets = batch["boneage"].to(
                    device,
                    dtype=torch.float32
                )

            else:

                images, targets = batch

                images = images.to(device)

                targets = targets.to(
                    device,
                    dtype=torch.float32
                )

            targets = targets.reshape(-1)

            outputs = model(images).reshape(-1)

            loss = criterion(outputs, targets)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size

            total_absolute_error += torch.abs(
                outputs - targets
            ).sum().item()

            total_samples += batch_size

            print(
                f"\rValidation batch "
                f"{batch_index}/{len(dataloader)}",
                end="",
                flush=True
            )

    print()

    validation_loss = running_loss / total_samples

    validation_mae = (
        total_absolute_error / total_samples
    )

    return validation_loss, validation_mae


# ==========================================================
# 8. COMPLETE TRAINING PIPELINE
# ==========================================================

def run_training_pipeline():

    set_seed()

    print("=" * 60)
    print("PEDIATRIC BONE AGE - RESNET50 BASELINE")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode enabled.")

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("\nMode: SMOKE TEST")
    print(f"Training limit: {TRAIN_LIMIT}")
    print(f"Validation limit: {VAL_LIMIT}")

    train_loader, val_loader = create_dataloaders()

    print("\nInitializing ResNet-50 model...")

    model = ResNet50Baseline(pretrained=True).to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_mae = float("inf")

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch + 1:02d}/{EPOCHS:02d}]")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_mae = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\nTraining MAE: {train_loss:.2f} months")
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Validation MAE: {val_mae:.2f} months")

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_mae": val_mae,
                    "image_size": IMAGE_SIZE
                },
                CHECKPOINT_PATH
            )

            print("\n--> Best model checkpoint saved!")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED")
    print("=" * 60)

    print(f"Best validation MAE: {best_val_mae:.2f} months")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    print(
        "\nNOTE: This result uses a subset of the dataset "
        "and is not the final research baseline."
    )

    return best_val_mae


# ==========================================================
# 9. ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_training_pipeline()
