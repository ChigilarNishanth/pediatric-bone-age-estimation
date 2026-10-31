import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Ensure project root and src directory are in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import apply_clahe_preprocessing

def run_eda(csv_path: str, img_dir: str, output_dir: str = "docs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Summary Statistics
    print(f"Total Radiographs: {len(df)}")
    print(f"Bone Age Range: {df['boneage'].min()} - {df['boneage'].max()} months")
    print(f"Mean Bone Age: {df['boneage'].mean():.2f} +/- {df['boneage'].std():.2f} months")
    print(f"Sex Distribution:\n{df['male'].value_counts(normalize=True) * 100}")

    # 2. Age Distribution & Sex Split Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['boneage'], kde=True, color='skyblue', bins=30)
    plt.title('RSNA Bone Age Distribution (Months)')
    plt.xlabel('Bone Age (Months)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sex_counts = df['male'].value_counts()
    plt.pie(sex_counts, labels=['Male (True)', 'Female (False)'], autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'])
    plt.title('Biological Sex Split')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rsna_eda_distributions.png"), dpi=300)
    plt.close()
    print(f"Saved EDA plot to {output_dir}/rsna_eda_distributions.png")

    # 3. OpenCV CLAHE Comparison Sample
    sample_row = df.iloc[0]
    sample_path = os.path.join(img_dir, f"{sample_row['id']}.png")
    if os.path.exists(sample_path):
        raw_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        clahe_img = apply_clahe_preprocessing(sample_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_img, cmap='gray')
        plt.title(f"Raw Radiograph (ID: {sample_row['id']})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(clahe_img)
        plt.title("OpenCV CLAHE (512x512, Clip: 2.0, Grid: 8x8)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clahe_comparison_sample.png"), dpi=300)
        plt.close()
        print(f"Saved CLAHE sample comparison to {output_dir}/clahe_comparison_sample.png")

if __name__ == "__main__":
    csv_path = "./data/boneage-training-dataset.csv"
    img_dir = "./data/boneage-training-dataset/boneage-training-dataset"
    if os.path.exists(csv_path):
        run_eda(csv_path, img_dir)
    else:
        print("Data files not found locally. Run this script once dataset is downloaded.")
