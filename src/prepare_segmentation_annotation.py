from pathlib import Path
import shutil
import pandas as pd

BASE = Path(r"D:\Pediatric_Bone_Age")
CSV_PATH = BASE / "RSNA_Dataset" / "Annotations" / "train.csv"
IMAGE_DIR = BASE / "RSNA_Dataset" / "train_images" / "boneage-training-dataset"
OUT = BASE / "segmentation_dataset"
SAMPLE_SIZE = 50
RANDOM_STATE = 42

def main():
    images_out = OUT / "images"
    masks_out = OUT / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].astype(int)
    df["age_group"] = pd.qcut(df["boneage"], q=4, labels=False, duplicates="drop")
    # Stratify jointly by sex and age quartile; proportional sampling.
    df["stratum"] = df["male"].astype(str) + "_" + df["age_group"].astype(str)
    sampled = (
        df.groupby("stratum", group_keys=False, observed=False)
          .apply(lambda group: group.sample(
              n=max(1, round(SAMPLE_SIZE * len(group) / len(df))),
              random_state=RANDOM_STATE
          ))
    )
    if len(sampled) > SAMPLE_SIZE:
        sampled = sampled.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    elif len(sampled) < SAMPLE_SIZE:
        remaining = df.loc[~df["id"].isin(sampled["id"])]
        sampled = pd.concat([sampled, remaining.sample(
            n=SAMPLE_SIZE - len(sampled), random_state=RANDOM_STATE
        )])

    rows = []
    for row in sampled.itertuples(index=False):
        image_id = int(row.id)
        source = IMAGE_DIR / f"{image_id}.png"
        target = images_out / f"{image_id}.png"
        status = "pending_annotation"
        if source.exists():
            shutil.copy2(source, target)
        else:
            status = "missing_image"
        rows.append({
            "id": image_id, "boneage": row.boneage, "male": row.male,
            "age_group": int(row.age_group), "image_path": str(target),
            "mask_path": str(masks_out / f"{image_id}_mask.png"),
            "annotation_status": status, "split": "pilot"
        })

    manifest = pd.DataFrame(rows, columns=[
        "id", "boneage", "male", "age_group", "image_path",
        "mask_path", "annotation_status", "split"
    ])
    manifest.to_csv(OUT / "annotation_manifest.csv", index=False)
    print(f"Selected: {len(manifest)}")
    print(f"Images copied: {(manifest.annotation_status == 'pending_annotation').sum()}")
    print(f"Missing images: {(manifest.annotation_status == 'missing_image').sum()}")
    print("Sex counts:")
    print(manifest["male"].value_counts(dropna=False).to_string())
    print("Age-group counts:")
    print(manifest["age_group"].value_counts().sort_index().to_string())
    print("Masks are placeholders only; no ground-truth masks were created.")

if __name__ == "__main__":
    main()
