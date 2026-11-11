from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE = Path(r"D:\Pediatric_Bone_Age")
CSV_PATH = BASE / "RSNA_Dataset" / "Annotations" / "train.csv"
IMAGE_DIR = BASE / "RSNA_Dataset" / "train_images" / "boneage-training-dataset"
OUT_IMAGES = BASE / "preprocessing" / "hybrid_processed_images"
OUT_ARRAYS = BASE / "preprocessing" / "hybrid_normalized_arrays"
LOG_DIR = BASE / "preprocessing" / "hybrid_logs"
SIZE = (512, 512)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
TOPHAT_KERNEL = 15
GAMMA = 1.0

def apply_gamma(image, gamma):
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)

def main():
    for folder in (OUT_IMAGES, OUT_ARRAYS, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    successful, failed, skipped = [], [], []
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL, TOPHAT_KERNEL))

    for row in df.itertuples(index=False):
        image_id = int(getattr(row, "id"))
        image_path = IMAGE_DIR / f"{image_id}.png"
        if not image_path.exists():
            skipped.append({"image_id": image_id, "reason": "missing_image"})
            continue
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("OpenCV could not read image")
            image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
            enhanced = clahe.apply(image)
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            combined = cv2.addWeighted(enhanced, 0.75, tophat, 0.25, 0)
            corrected = apply_gamma(combined, GAMMA)
            normalized = corrected.astype(np.float32) / 255.0
            mean, std = float(normalized.mean()), float(normalized.std())
            zscore = (normalized - mean) / (std if std > 1e-8 else 1.0)

            cv2.imwrite(str(OUT_IMAGES / f"{image_id}.png"), corrected)
            np.save(OUT_ARRAYS / f"{image_id}.npy", zscore.astype(np.float32))
            successful.append({
                "image_id": image_id, "mean": mean, "std": std,
                "gamma": GAMMA, "width": SIZE[0], "height": SIZE[1]
            })
        except Exception as exc:
            failed.append({"image_id": image_id, "reason": str(exc)})

    pd.DataFrame(successful).to_csv(LOG_DIR / "sample_hybrid_successful.csv", index=False)
    pd.DataFrame(failed, columns=["image_id", "reason"]).to_csv(LOG_DIR / "sample_hybrid_failed.csv", index=False)
    pd.DataFrame(skipped, columns=["image_id", "reason"]).to_csv(LOG_DIR / "sample_hybrid_skipped.csv", index=False)
    print(f"Processed: {len(successful)} | Failed: {len(failed)} | Skipped: {len(skipped)}")

if __name__ == "__main__":
    main()
