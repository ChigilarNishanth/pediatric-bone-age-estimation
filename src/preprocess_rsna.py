import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. DATASET PATHS
# ==========================================

CSV_PATH = r"D:\Pediatric_Bone_Age\RSNA_Dataset\Annotations\train.csv"

IMAGE_DIR = r"D:\Pediatric_Bone_Age\RSNA_Dataset\train_images\boneage-training-dataset"

OUTPUT_DIR = r"D:\Pediatric_Bone_Age\preprocessing\processed_images"

LOG_DIR = r"D:\Pediatric_Bone_Age\preprocessing\logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 2. CLAHE CONFIGURATION
# ==========================================

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
OUTPUT_SIZE = (512, 512)

# ==========================================
# 3. IMAGE PREPROCESSING FUNCTION
# ==========================================

def preprocess_xray(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image")

    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE
    )

    enhanced = clahe.apply(image)

    # Preserve aspect ratio
    target_w, target_h = OUTPUT_SIZE
    h, w = enhanced.shape

    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        enhanced,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center padding
    canvas = np.zeros(
        (target_h, target_w),
        dtype=np.uint8
    )

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ==========================================
# 4. LOAD METADATA
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Total training records:", len(df))
print("Starting CLAHE preprocessing...")


# ==========================================
# 5. PROCESS ALL IMAGES
# ==========================================

successful = []
failed = []

for image_id in tqdm(
    df["id"].astype(str),
    desc="CLAHE Processing"
):

    input_path = os.path.join(
        IMAGE_DIR,
        image_id + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        image_id + ".png"
    )

    try:

        processed = preprocess_xray(input_path)

        success = cv2.imwrite(
            output_path,
            processed
        )

        if not success:
            raise IOError("Unable to save processed image")

        successful.append(image_id)

    except Exception as error:

        failed.append({
            "id": image_id,
            "error": str(error)
        })


# ==========================================
# 6. SAVE PROCESSING LOGS
# ==========================================

pd.DataFrame({
    "image_id": successful
}).to_csv(
    os.path.join(LOG_DIR, "successful_images.csv"),
    index=False
)

pd.DataFrame(failed).to_csv(
    os.path.join(LOG_DIR, "failed_images.csv"),
    index=False
)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING SUMMARY ==========")

print("Total images:", len(df))
print("Successfully processed:", len(successful))
print("Failed images:", len(failed))

print("CLAHE Clip Limit:", CLIP_LIMIT)
print("CLAHE Tile Grid:", TILE_GRID_SIZE)
print("Output Size:", OUTPUT_SIZE)

print("Output Directory:", OUTPUT_DIR)

print("\nPreprocessing completed!")