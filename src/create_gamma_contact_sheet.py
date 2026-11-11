from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(r"D:\Pediatric_Bone_Age")
ROOT = BASE / "preprocessing" / "gamma_comparison"
METRICS = ROOT / "gamma_comparison_metrics.csv"
GAMMAS = ("0.8", "1.0", "1.2")
THUMB = (256, 256)
IMAGES_PER_SHEET = 5

def find_image(image_id, gamma):
    candidates = [
        ROOT / f"{image_id}_gamma_{gamma}.png",
        ROOT / f"{image_id}_gamma_{float(gamma):.1f}.png",
        ROOT / "images" / f"{image_id}_gamma_{gamma}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

def main():
    df = pd.read_csv(METRICS)
    id_col = "image_id" if "image_id" in df.columns else "id"
    ids = df[id_col].dropna().astype(int).drop_duplicates().tolist()
    panels = []
    for image_id in ids:
        row_panels = []
        for gamma in GAMMAS:
            path = find_image(image_id, gamma)
            if path is None:
                continue
            image = Image.open(path).convert("L").resize(THUMB)
            cell = Image.new("RGB", (THUMB[0] + 20, THUMB[1] + 48), "white")
            cell.paste(image.convert("RGB"), (10, 32))
            ImageDraw.Draw(cell).text((10, 8), f"ID {image_id} | gamma {gamma}", fill="black")
            row_panels.append(cell)
        if row_panels:
            panels.append(row_panels)

    ROOT.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(panels), IMAGES_PER_SHEET):
        batch = panels[start:start + IMAGES_PER_SHEET]
        sheet = Image.new("RGB", (3 * (THUMB[0] + 20), len(batch) * (THUMB[1] + 48)), "white")
        for row_index, row_panels in enumerate(batch):
            for col_index, panel in enumerate(row_panels):
                sheet.paste(panel, (col_index * (THUMB[0] + 20), row_index * (THUMB[1] + 48)))
        sheet.save(ROOT / f"gamma_contact_sheet_{start // IMAGES_PER_SHEET + 1}.png")
    print(f"Created {(len(panels) + IMAGES_PER_SHEET - 1) // IMAGES_PER_SHEET} contact sheets for {len(panels)} image IDs.")

if __name__ == "__main__":
    main()
