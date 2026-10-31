import cv2
import matplotlib.pyplot as plt

original_path = r"D:\Pediatric_Bone_Age\RSNA_Dataset\train_images\boneage-training-dataset\1377.png"

processed_path = r"D:\Pediatric_Bone_Age\preprocessing\processed_images\1377.png"

original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed, cmap="gray")
plt.title("CLAHE Preprocessed X-ray")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    r"D:\Pediatric_Bone_Age\preprocessing\logs\original_vs_clahe.png",
    dpi=300
)

plt.show()