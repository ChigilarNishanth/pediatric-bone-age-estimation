from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(r"D:\Pediatric_Bone_Age")
ORIGINAL_DIR = BASE / "RSNA_Dataset" / "train_images" / "boneage-training-dataset"
HYBRID_DIR = BASE / "preprocessing" / "hybrid_processed_images"
METRICS_SOURCE = BASE / "preprocessing" / "gamma_comparison" / "gamma_comparison_metrics.csv"
OUT = BASE / "preprocessing" / "otsu_segmentation"
SHEET_SIZE = 5

def make_mask(image_path, out_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot read image")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(out_path), mask)
    foreground = float(np.count_nonzero(mask)) / mask.size * 100.0
    return threshold, foreground, image.shape[1], image.shape[0]

def contact_sheets(records, output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), SHEET_SIZE):
        batch = records[start:start + SHEET_SIZE]
        thumbs = []
        for record in batch:
            im = Image.open(record["path"]).convert("L").resize((256, 256))
            canvas = Image.new("RGB", (276, 292), "white")
            canvas.paste(im.convert("RGB"), (10, 26))
            ImageDraw.Draw(canvas).text((10, 7), str(record["image_id"]), fill="black")
            thumbs.append(canvas)
        sheet = Image.new("RGB", (276 * len(thumbs), 292), "white")
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, (276 * i, 0))
        sheet.save(output_dir / f"{prefix}_{start // SHEET_SIZE + 1}.png")

def run_group(ids, source_dir, group_name):
    mask_dir = OUT / group_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows, sheet_records = [], []
    for image_id in ids:
        src = source_dir / f"{image_id}.png"
        if not src.exists():
            continue
        dst = mask_dir / f"{image_id}_otsu.png"
        try:
            threshold, pct, width, height = make_mask(src, dst)
            rows.append({"image_id": image_id, "threshold": threshold,
                         "foreground_percentage": pct, "image_width": width,
                         "image_height": height})
            sheet_records.append({"image_id": image_id, "path": dst})
        except Exception as exc:
            print(f"{group_name} {image_id}: {exc}")
    pd.DataFrame(rows, columns=["image_id", "threshold", "foreground_percentage",
                                "image_width", "image_height"]).to_csv(
        OUT / f"{group_name}_otsu_metrics.csv", index=False)
    contact_sheets(sheet_records, OUT / "contact_sheets", f"{group_name}_otsu")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_SOURCE)
    id_col = "image_id" if "image_id" in metrics.columns else "id"
    ids = metrics[id_col].dropna().astype(int).drop_duplicates().head(20).tolist()
    run_group(ids, ORIGINAL_DIR, "original")
    run_group(ids, HYBRID_DIR, "hybrid")
    print(f"Otsu comparison complete for {len(ids)} sample IDs.")

if __name__ == "__main__":
    main()
