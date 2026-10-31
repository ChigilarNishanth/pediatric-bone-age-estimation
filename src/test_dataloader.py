from pathlib import Path
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from rsna_dataset import RSNABoneAgeDataset


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

# Repository root
BASE = Path(__file__).resolve().parents[1]

# Metadata stored inside the repository
TRAIN_METADATA = BASE / "metadata" / "train_metadata.csv"
VAL_METADATA = BASE / "metadata" / "validation_metadata.csv"

# Directory containing processed_images and
# processed_validation_images.
# Teammates can configure their own location using
# the RSNA_DATA_ROOT environment variable.
DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        r"D:\Pediatric_Bone_Age\preprocessing"
    )
)


# ==================================================
# 2. IMAGE TRANSFORMATION
# ==================================================

transform = transforms.ToTensor()


# ==================================================
# 3. TRAINING DATASET TEST
# ==================================================

print("\n========== TRAINING DATALOADER TEST ==========")

train_dataset = RSNABoneAgeDataset(
    metadata_csv=TRAIN_METADATA,
    transform=transform,
    data_root=DATA_ROOT
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

train_batch = next(iter(train_loader))

print("Training samples:", len(train_dataset))
print("Image batch shape:", train_batch["image"].shape)
print("Bone age shape:", train_batch["boneage"].shape)
print("Sex shape:", train_batch["male"].shape)
print("Image IDs:", train_batch["image_id"])

print("Training DataLoader test passed!")


# ==================================================
# 4. VALIDATION DATASET TEST
# ==================================================

print("\n========== VALIDATION DATALOADER TEST ==========")

validation_dataset = RSNABoneAgeDataset(
    metadata_csv=VAL_METADATA,
    transform=transform,
    data_root=DATA_ROOT
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

validation_batch = next(iter(validation_loader))

print("Validation samples:", len(validation_dataset))
print("Image batch shape:", validation_batch["image"].shape)
print("Bone age shape:", validation_batch["boneage"].shape)
print("Sex shape:", validation_batch["male"].shape)
print("Image IDs:", validation_batch["image_id"])

print("Validation DataLoader test passed!")


# ==================================================
# 5. FINAL RESULT
# ==================================================

print("\n======================================")
print("ALL DATALOADER TESTS PASSED!")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Total samples:", len(train_dataset) + len(validation_dataset))
print("======================================")