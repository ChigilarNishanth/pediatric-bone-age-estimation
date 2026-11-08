import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RSNABoneAgeMultiVariantDataset(Dataset):
    """
    Multi-Variant PyTorch Dataset for RSNA Pediatric Bone Age Assessment.
    Supports comparative benchmarking across four preprocessing pipelines:
      - 'raw': PA radiographs without enhancement
      - 'clahe': Baseline OpenCV CLAHE contrast tuning
      - 'hybrid': CLAHE + Top-Hat + Gamma Correction + Z-Score
      - 'unet_masked': U-Net background-segmented & cropped hand outlines
    """
    VALID_VARIANTS = {'raw', 'clahe', 'hybrid', 'unet_masked'}

    def __init__(self, csv_file, base_data_dir, variant='clahe', transform=None, img_size=(512, 512)):
        self.df = pd.read_csv(csv_file)
        self.base_data_dir = base_data_dir
        self.variant = variant.lower()
        self.img_size = img_size

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(f"Invalid variant '{self.variant}'. Must be one of {self.VALID_VARIANTS}")

        # Path resolution for target variant folder
        self.variant_dir = os.path.join(self.base_data_dir, self.variant)
        if not os.path.exists(self.variant_dir):
            raise FileNotFoundError(f"Variant directory missing: {self.variant_dir}")

        # Default PyTorch evaluation pipeline calibrated for pretrained ResNet-50
        if transform is None:
            self.transform = T.Compose([
                T.Resize(self.img_size),
                T.ToTensor(),  # Scales PIL image [0, 255] to Tensor [0.0, 1.0] (C, H, W)
                # ResNet-50 requires 3 channels; expand 1-channel grayscale tensors
                T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['id'])
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.variant_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image for variant '{self.variant}' missing: {img_path}")

        # Load in 8-bit grayscale mode to standardize intensity ranges
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        # Bone age in continuous months (Float32 target for MAE regression)
        bone_age = torch.tensor(row['boneage'], dtype=torch.float32)
        
        # Biological sex encoding (1.0 = Male, 0.0 = Female)
        is_male = torch.tensor(1.0 if row['male'] else 0.0, dtype=torch.float32)

        return {
            'image': image,
            'bone_age': bone_age,
            'is_male': is_male,
            'img_id': img_id
        }