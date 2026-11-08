import os
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image

# Multi-variant configuration and checkpoint paths
VALID_VARIANTS = {'raw', 'clahe', 'hybrid', 'unet_masked'}
CHECKPOINT_DIR = "checkpoints"

def load_resnet50_model(variant='clahe', device=None):
    """
    Loads ResNet-50 evaluation checkpoint for target preprocessing variant.
    Falls back to resnet50_baseline_best.pth if variant checkpoint is missing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"resnet50_{variant}_best.pth")
    fallback_path = os.path.join(CHECKPOINT_DIR, "resnet50_baseline_best.pth")

    target_path = checkpoint_path if os.path.exists(checkpoint_path) else fallback_path
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path} or {fallback_path}")

    checkpoint = torch.load(target_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def transform_xray_image(image_input, img_size=(512, 512)):
    """Transforms PIL Image or file path into a 3-channel normalized tensor."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('L')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('L')
    else:
        raise ValueError("Invalid image input type. Provide a file path or PIL.Image instance.")

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_bone_age(image_input, variant='clahe', device=None):
    """
    Primary API endpoint for Mahesh's Streamlit frontend.
    
    Args:
        image_input (str | PIL.Image): X-ray image file path or loaded PIL image.
        variant (str): Selected preprocessing variant ('raw', 'clahe', 'hybrid', 'unet_masked').
        device (torch.device, optional): Execution device target.
        
    Returns:
        dict: Summary containing predicted age in months, years, and variant used.
    """
    variant_clean = variant.lower()
    if variant_clean not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Supported options: {VALID_VARIANTS}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_resnet50_model(variant=variant_clean, device=device)
    input_tensor = transform_xray_image(image_input).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return {
        "predicted_age_months": round(float(prediction), 2),
        "predicted_age_years": round(float(prediction) / 12.0, 2),
        "variant_used": variant_clean,
        "status": "SUCCESS"
    }