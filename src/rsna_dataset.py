from pathlib import Path
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class RSNABoneAgeDataset(Dataset):

    def __init__(self, metadata_csv, transform=None, data_root=None):

        self.metadata_csv = Path(metadata_csv).resolve()
        self.data = pd.read_csv(self.metadata_csv)
        self.transform = transform

        self.data_root = Path(
            data_root or os.environ.get(
                "RSNA_DATA_ROOT",
                str(self.metadata_csv.parent.parent / "data")
            )
        ).resolve()

        required_columns = [
            "image_id",
            "image_path",
            "boneage",
            "male",
            "split"
        ]

        missing = set(required_columns) - set(self.data.columns)

        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        self.image_paths = []

        for relative_path in self.data["image_path"]:
            image_path = (self.data_root / relative_path).resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Processed image not found: {image_path}"
                )

            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        with Image.open(self.image_paths[index]) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "boneage": torch.tensor(
                float(row["boneage"]), dtype=torch.float32
            ),
            "male": torch.tensor(
                float(row["male"]), dtype=torch.float32
            ),
            "image_id": int(row["image_id"])
        }