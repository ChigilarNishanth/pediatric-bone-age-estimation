from pathlib import Path
import os
import pandas as pd


# ==================================================
# 1. PROJECT PATH CONFIGURATION
# ==================================================

# Repository root (parent of src folder)
BASE = Path(__file__).resolve().parents[1]

# Dataset root on your computer.
# Teammates can set RSNA_PROJECT_ROOT to their own location.
PROJECT_ROOT = Path(
    os.environ.get(
        "RSNA_PROJECT_ROOT",
        r"D:\Pediatric_Bone_Age"
    )
).resolve()


# ==================================================
# 2. INPUT FILE PATHS
# ==================================================

# Original training annotations
TRAIN_CSV = (
    PROJECT_ROOT
    / "RSNA_Dataset"
    / "Annotations"
    / "train.csv"
)

# Original validation annotations
VAL_CSV = (
    PROJECT_ROOT
    / "RSNA_Dataset"
    / "validation_extracted"
    / "Bone Age Validation Set"
    / "Validation Dataset.csv"
)


# ==================================================
# 3. PROCESSED IMAGE DIRECTORIES
# ==================================================

TRAIN_IMAGES = (
    PROJECT_ROOT
    / "preprocessing"
    / "processed_images"
)

VAL_IMAGES = (
    PROJECT_ROOT
    / "preprocessing"
    / "processed_validation_images"
)


# ==================================================
# 4. OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR = BASE / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 5. CHECK INPUT DIRECTORIES AND FILES
# ==================================================

required_paths = [
    TRAIN_CSV,
    VAL_CSV,
    TRAIN_IMAGES,
    VAL_IMAGES
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset path not found: {path}"
        )


# ==================================================
# 6. PREPARE TRAINING / VALIDATION METADATA
# ==================================================

def prepare_split(
    csv_path,
    image_dir,
    split_name,
    id_column,
    age_column
):

    df = pd.read_csv(csv_path)

    # Standardize annotation column names
    df = df.rename(
        columns={
            id_column: "image_id",
            age_column: "boneage"
        }
    )

    # Convert data types
    df["image_id"] = df["image_id"].astype(int)
    df["boneage"] = pd.to_numeric(
        df["boneage"],
        errors="raise"
    )
    df["male"] = df["male"].astype(int)

    # Create grouping key.
    # This is a provisional image-ID-based grouping key,
    # not an independently verified patient identifier.
    df["patient_group_id"] = (
        df["image_id"].astype(str)
    )

    # Store portable relative image paths
    df["image_path"] = df["image_id"].apply(
        lambda image_id: (
            Path(image_dir.name) / f"{image_id}.png"
        ).as_posix()
    )

    df["split"] = split_name

    # Check that all processed images exist
    missing_images = [
        image_id
        for image_id in df["image_id"]
        if not (image_dir / f"{image_id}.png").is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} {split_name} images "
            f"are missing.\n"
            f"First missing IDs: {missing_images[:10]}"
        )

    # Keep consistent metadata columns
    columns = [
        "image_id",
        "patient_group_id",
        "image_path",
        "boneage",
        "male",
        "split"
    ]

    return df[columns]


# ==================================================
# 7. GENERATE TRAINING METADATA
# ==================================================

print("\nPreparing training metadata...")

train = prepare_split(
    csv_path=TRAIN_CSV,
    image_dir=TRAIN_IMAGES,
    split_name="train",
    id_column="id",
    age_column="boneage"
)


# ==================================================
# 8. GENERATE VALIDATION METADATA
# ==================================================

print("\nPreparing validation metadata...")

val = prepare_split(
    csv_path=VAL_CSV,
    image_dir=VAL_IMAGES,
    split_name="validation",
    id_column="Image ID",
    age_column="Bone Age (months)"
)


# ==================================================
# 9. DATA QUALITY CHECKS
# ==================================================

# Check duplicate image IDs
assert not train["image_id"].duplicated().any(), (
    "Duplicate training image IDs found!"
)

assert not val["image_id"].duplicated().any(), (
    "Duplicate validation image IDs found!"
)

# Check sex values
assert train["male"].isin([0, 1]).all(), (
    "Invalid training sex values found!"
)

assert val["male"].isin([0, 1]).all(), (
    "Invalid validation sex values found!"
)

# Check bone-age range
assert train["boneage"].between(0, 228).all(), (
    "Training bone age outside expected range!"
)

assert val["boneage"].between(0, 228).all(), (
    "Validation bone age outside expected range!"
)

# Check train-validation ID overlap
train_ids = set(train["image_id"])
val_ids = set(val["image_id"])

overlap = train_ids.intersection(val_ids)

assert len(overlap) == 0, (
    f"Train-validation ID overlap found: {len(overlap)}"
)


# ==================================================
# 10. SAVE METADATA CSV FILES
# ==================================================

train_output = OUTPUT_DIR / "train_metadata.csv"

val_output = OUTPUT_DIR / "validation_metadata.csv"

train.to_csv(train_output, index=False)

val.to_csv(val_output, index=False)


# ==================================================
# 11. FINAL SUMMARY
# ==================================================

print("\n======================================")
print("       METADATA PREPARATION SUMMARY")
print("======================================")

print("Training records:", len(train))
print("Validation records:", len(val))

print("Duplicate training IDs:", train["image_id"].duplicated().sum())
print("Duplicate validation IDs:", val["image_id"].duplicated().sum())

print("Invalid training sex values:",
      (~train["male"].isin([0, 1])).sum())

print("Invalid validation sex values:",
      (~val["male"].isin([0, 1])).sum())

print("Missing training images: 0")
print("Missing validation images: 0")

print("Train-validation ID overlap:", len(overlap))

print("\nTraining metadata saved to:")
print(train_output)

print("\nValidation metadata saved to:")
print(val_output)

print("\nTraining metadata sample:")
print(train.head(3).to_string(index=False))

print("\nValidation metadata sample:")
print(val.head(3).to_string(index=False))

print("\nMetadata preparation completed successfully!")