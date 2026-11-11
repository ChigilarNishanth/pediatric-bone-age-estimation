from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(r"D:\Pediatric_Bone_Age")
IMAGE_DIR = BASE / "RSNA_Dataset" / "train_images" / "boneage-training-dataset"
METRICS_SOURCE = BASE / "preprocessing" / "gamma_comparison" / "gamma_comparison_metrics.csv"
OUT = BASE / "preprocessing" / "otsu_refined"
MIN_AREA_RATIO = 0.001
BORDER_MARGIN = 2

def make_refined_mask(image_path, mask_path, overlay_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot read image")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    threshold, raw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel_close)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * MIN_AREA_RATIO
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    selected = max(candidates, key=cv2.contourArea) if candidates else None
    final_mask = np.zeros_like(image)
    touches_border = False
    largest_area = 0.0
    if selected is not None:
        largest_area = float(cv2.contourArea(selected))
        cv2.drawContours(final_mask, [selected], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(selected)
        touches_border = (
            x <= BORDER_MARGIN or y <= BORDER_MARGIN or
            x + w >= image.shape[1] - BORDER_MARGIN or
            y + h >= image.shape[0] - BORDER_MARGIN
        )
    cv2.imwrite(str(mask_path), final_mask)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[final_mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "threshold": threshold, "candidate_components": len(candidates),
        "largest_component_area": largest_area,
        "foreground_percentage": float(np.count_nonzero(final_mask)) / final_mask.size * 100,
        "mask_found": selected is not None,
        "selected_contour_touches_border": touches_border
    }

def create_sheets(items, folder, prefix):
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), 5):
        batch = items[start:start + 5]
        cells = []
        for image_id, path in batch:
            im = Image.open(path).convert("RGB").resize((256, 256))
            cell = Image.new("RGB", (276, 282), "white")
            cell.paste(im, (10, 22))
            ImageDraw.Draw(cell).text((10, 5), str(image_id), fill="black")
            cells.append(cell)
        sheet = Image.new("RGB", (276 * len(cells), 282), "white")
        for i, cell in enumerate(cells):
            sheet.paste(cell, (i * 276, 0))
        sheet.save(folder / f"{prefix}_{start // 5 + 1}.png")

def main():
    for name in ("raw_masks", "refined_masks", "overlays", "contact_sheets"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_SOURCE)
    id_col = "image_id" if "image_id" in metrics.columns else "id"
    ids = metrics[id_col].dropna().astype(int).drop_duplicates().head(20).tolist()
    rows, mask_items, overlay_items, failures = [], [], [], []
    for image_id in ids:
        src = IMAGE_DIR / f"{image_id}.png"
        if not src.exists():
            failures.append({"image_id": image_id, "reason": "missing_image"})
            continue
        raw_path = OUT / "raw_masks" / f"{image_id}_raw.png"
        refined_path = OUT / "refined_masks" / f"{image_id}_refined.png"
        overlay_path = OUT / "overlays" / f"{image_id}_overlay.png"
        try:
            result = make_refined_mask(src, refined_path, overlay_path)
            # Save the threshold-only mask separately for visual comparison.
            image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            _, raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(str(raw_path), raw)
            rows.append({"image_id": image_id, **result})
            mask_items.append((image_id, refined_path))
            overlay_items.append((image_id, overlay_path))
        except Exception as exc:
            failures.append({"image_id": image_id, "reason": str(exc)})
    columns = ["image_id", "threshold", "candidate_components", "largest_component_area",
               "foreground_percentage", "mask_found", "selected_contour_touches_border"]
    pd.DataFrame(rows, columns=columns).to_csv(OUT / "refined_otsu_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["image_id", "reason"]).to_csv(OUT / "refined_otsu_failures.csv", index=False)
    create_sheets(mask_items, OUT / "contact_sheets", "refined_otsu")
    create_sheets(overlay_items, OUT / "contact_sheets", "refined_overlay")
    print(f"Attempted {len(ids)} | Processed {len(rows)} | Failed {len(failures)}")
    if rows:
        print(f"Average foreground: {np.mean([r['foreground_percentage'] for r in rows]):.2f}%")
        border_count = sum(bool(r["selected_contour_touches_border"]) for r in rows)
        print(f"Contours touching border: {border_count}/{len(rows)}")

if __name__ == "__main__":
    main()
