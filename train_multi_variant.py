import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from dataset import RSNABoneAgeMultiVariantDataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['bone_age'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_mae = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['bone_age'].to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_mae += loss.item() * images.size(0)
    return running_mae / len(dataloader.dataset)

def run_variant_experiment(variant, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting Experiment: Variant [{variant.upper()}] ---")

    train_dataset = RSNABoneAgeMultiVariantDataset("data/train.csv", "data/", variant=variant)
    val_dataset = RSNABoneAgeMultiVariantDataset("data/valid.csv", "data/", variant=variant)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    writer = SummaryWriter(log_dir=f"runs/variant_{variant}")
    history = {"train_loss": [], "val_mae": []}

    best_mae = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mae = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("MAE/Validation", val_mae, epoch)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.2f} | Val MAE: {val_mae:.2f} Mos")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_mae": val_mae},
                f"checkpoints/resnet50_{variant}_best.pth"
            )

    writer.close()
    
    with open(f"checkpoints/history_{variant}.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"Variant [{variant.upper()}] Finished. Best Val MAE: {best_mae:.2f} Months")
    return history

if __name__ == "__main__":
    variants = ['raw', 'clahe', 'hybrid', 'unet_masked']
    for v in variants:
        run_variant_experiment(variant=v, epochs=10)