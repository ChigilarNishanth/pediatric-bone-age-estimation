import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
from dataset import RSNABoneAgeMultiVariantDataset

def load_resnet50_baseline(checkpoint_path, device):
    """Loads frozen ResNet-50 baseline model for continuous age regression."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    # Single continuous regression output head for continuous age prediction
    model.fc = nn.Linear(in_features, 1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

def evaluate_variant(model, dataloader, device):
    """Evaluates MAE performance for a specific input variant stream."""
    total_mae = 0.0
    total_samples = 0
    criterion = nn.L1Loss(reduction='sum')

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['bone_age'].to(device).unsqueeze(1)

            predictions = model(images)
            loss = criterion(predictions, targets)

            total_mae += loss.item()
            total_samples += targets.size(0)

    return total_mae / total_samples

def run_comparative_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Multi-Variant Benchmark on Device: {device}")

    CSV_FILE = "data/valid.csv"
    BASE_DIR = "data/"
    CHECKPOINT_PATH = "checkpoints/resnet50_baseline_best.pth"

    model = load_resnet50_baseline(CHECKPOINT_PATH, device)
    variants = ['raw', 'clahe', 'hybrid', 'unet_masked']
    results = {}

    for variant in variants:
        dataset = RSNABoneAgeMultiVariantDataset(
            csv_file=CSV_FILE,
            base_data_dir=BASE_DIR,
            variant=variant
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        mae = evaluate_variant(model, dataloader, device)
        results[variant] = mae
        print(f"Variant: [{variant.upper():<11}] | Validation MAE: {mae:.2f} Months")

    # Output Summary Table
    df_res = pd.DataFrame(list(results.items()), columns=['Preprocessing Variant', 'Validation MAE (Months)'])
    print("\n--- Review 3 Comparative Benchmark Results ---")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_comparative_benchmark()