import cv2
import numpy as np
import os

def apply_tophat_filter(img, kernel_size=(15, 15)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return cv2.add(img, tophat)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def apply_gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_zscore_normalization(img):
    img_float = img.astype(np.float32)
    mean, std = np.mean(img_float), np.std(img_float)
    if std == 0:
        return img
    normalized = (img_float - mean) / std
    rescaled = cv2.normalize(normalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return rescaled.astype(np.uint8)

def process_hybrid_pipeline(input_image_path, gamma=0.9):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {input_image_path}")
    img_resized = cv2.resize(img, (512, 512))
    img_tophat = apply_tophat_filter(img_resized)
    img_clahe = apply_clahe(img_tophat, clip_limit=2.0, tile_grid_size=(8, 8))
    img_gamma = apply_gamma_correction(img_clahe, gamma=gamma)
    return apply_zscore_normalization(img_gamma)

def batch_generate_hybrid_variant(input_dir, output_dir, gamma=0.9):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[Pavan Pipeline] Processing {len(files)} images to Hybrid Preprocessing variant...")
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, process_hybrid_pipeline(in_path, gamma=gamma))
    print(f"[Pavan Pipeline] Hybrid variant successfully saved to '{output_dir}'.")

if __name__ == "__main__":
    batch_generate_hybrid_variant("data/raw", "data/hybrid")
