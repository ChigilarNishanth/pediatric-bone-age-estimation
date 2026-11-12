import os
import torch
import torch.nn as nn
import cv2
import numpy as np

class UNetHandSegmentor(nn.Module):
    def __init__(self):
        super(UNetHandSegmentor, self).__init__()
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = double_conv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = double_conv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = double_conv(64, 32)
        self.final_conv = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.final_conv(d1))

def generate_unet_masked_image(image_path, model=None, device='cpu'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img_resized = cv2.resize(img, (512, 512))
    if model is None:
        _, binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        model.eval()
        tensor_in = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_in = tensor_in.to(device)
        with torch.no_grad():
            pred_mask = model(tensor_in).squeeze().cpu().numpy()
        mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return cv2.bitwise_and(img_resized, img_resized, mask=mask.astype(np.uint8))

def batch_generate_unet_variant(input_dir, output_dir, checkpoint_path=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = UNetHandSegmentor().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[Pavan Pipeline] Segmenting {len(files)} images to UNET_MASKED variant...")
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, generate_unet_masked_image(in_path, model=model, device=device))
    print(f"[Pavan Pipeline] U-Net masked variant saved to '{output_dir}'.")

if __name__ == "__main__":
    batch_generate_unet_variant("data/raw", "data/unet_masked")
