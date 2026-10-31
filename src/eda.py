import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

# Ensure repository root is in sys.path for clean imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Dynamic fallback imports for CLAHE preprocessing function
try:
    from src.preprocess import apply_clahe_preprocessing
except ModuleNotFoundError:
    try:
        from src.preprocess_rsna import apply_clahe_preprocessing
    except ModuleNotFoundError:
        def apply_clahe_preprocessing(image_path: str, target_size: tuple = (512, 512)):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

def find_metadata_and_image_dir():
    """Locates metadata CSV and image directory across common project layouts."""
    possible_csvs = [
        "./data/boneage-training-dataset.csv",
        "./metadata/train_metadata.csv",
        "./data/train_metadata.csv",
        "boneage-training-dataset.csv"
    ]
    possible_img_dirs = [
        "./data/boneage-training-dataset/boneage-training-dataset",
        "./data/boneage-training-dataset",
        "./data/images",
        "./data"
    ]

    selected_csv = next((c for c in possible_csvs if os.path.exists(c)), None)
    selected_img_dir = next((d for d in possible_img_dirs if os.path.exists(d)), None)

    return selected_csv, selected_img_dir

def run_eda(csv_path: str = None, img_dir: str = None, output_dir: str = "docs"):
    """Performs Exploratory Data Analysis and generates distribution & CLAHE sample artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    if not csv_path or not os.path.exists(csv_path) or not img_dir or not os.path.exists(img_dir):
        auto_csv, auto_img_dir = find_metadata_and_image_dir()
        csv_path = csv_path if (csv_path and os.path.exists(csv_path)) else auto_csv
        img_dir = img_dir if (img_dir and os.path.exists(img_dir)) else auto_img_dir

    if not csv_path or not os.path.exists(csv_path):
        print("⚠️ [EDA Warning] Metadata CSV not found. Please ensure RSNA dataset is downloaded inside ./data or ./metadata")
        return

    df = pd.read_csv(csv_path)
    
    # Standardize column name detection
    id_col = 'id' if 'id' in df.columns else ('image_id' if 'image_id' in df.columns else df.columns[0])
    age_col = 'boneage' if 'boneage' in df.columns else ('age' if 'age' in df.columns else df.columns[1])
    sex_col = 'male' if 'male' in df.columns else ('sex' if 'sex' in df.columns else None)

    print("==================================================")
    print(" RSNA PEDIATRIC BONE AGE DATASET - EDA REPORT")
    print("==================================================")
    print(f"Total Radiographs Analyzed : {len(df)}")
    print(f"Bone Age Range             : {df[age_col].min()} to {df[age_col].max()} months")
    print(f"Mean Bone Age ± Std        : {df[age_col].mean():.2f} ± {df[age_col].std():.2f} months")
    
    if sex_col:
        male_pct = (df[sex_col].sum() / len(df)) * 100
        print(f"Biological Sex Ratio       : {male_pct:.1f}% Male / {100 - male_pct:.1f}% Female")
    print("==================================================\n")

    # 1. Plot & Save Age Distribution Histogram + Biological Sex Pie Chart
    fig, axes = plt.subplots(1, 2 if sex_col else 1, figsize=(12, 5))
    
    ax_hist = axes[0] if sex_col else axes
    sns.histplot(df[age_col], kde=True, color='#2b5c8f', bins=30, ax=ax_hist)
    ax_hist.set_title('RSNA Bone Age Distribution (Months)', fontsize=12, fontweight='bold')
    ax_hist.set_xlabel('Bone Age (Months)', fontsize=10)
    ax_hist.set_ylabel('Frequency / Count', fontsize=10)
    ax_hist.grid(True, linestyle='--', alpha=0.5)

    if sex_col:
        sex_counts = df[sex_col].value_counts()
        labels = ['Male' if k in [True, 1, 'M', 'Male'] else 'Female' for k in sex_counts.index]
        axes[1].pie(sex_counts, labels=labels, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90, explode=(0.05, 0))
        axes[1].set_title('Biological Sex Split', fontsize=12, fontweight='bold')

    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "rsna_eda_distributions.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f" Saved distribution plot to: {dist_plot_path}")

    # 2. Plot & Save CLAHE Visual Comparison Sample
    if img_dir and os.path.exists(img_dir):
        sample_row = df.iloc[0]
        sample_id = sample_row[id_col]
        
        possible_sample_paths = [
            os.path.join(img_dir, f"{sample_id}.png"),
            os.path.join(img_dir, f"{sample_id}.jpg"),
            os.path.join(img_dir, f"{sample_id}")
        ]
        
        sample_path = next((p for p in possible_sample_paths if os.path.exists(p)), None)

        if sample_path:
            raw_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            clahe_img = apply_clahe_preprocessing(sample_path)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(raw_img, cmap='gray')
            plt.title(f"Raw PA Radiograph (ID: {sample_id})", fontsize=11, fontweight='bold')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(clahe_img)
            plt.title("OpenCV CLAHE Enhanced (512x512, Clip: 2.0, Tile: 8x8)", fontsize=11, fontweight='bold')
            plt.axis('off')

            plt.tight_layout()
            clahe_plot_path = os.path.join(output_dir, "clahe_comparison_sample.png")
            plt.savefig(clahe_plot_path, dpi=300)
            plt.close()
            print(f" Saved CLAHE sample comparison plot to: {clahe_plot_path}")

if __name__ == "__main__":
    run_eda()
