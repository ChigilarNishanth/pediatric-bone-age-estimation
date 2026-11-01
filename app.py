import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.baseline_resnet50 import ResNet50Baseline

st.set_page_config(page_title="Pediatric Bone Age Estimation", layout="centered")

st.title("Pediatric Bone Age & Sex Estimation")
st.write("Upload a postero-anterior (PA) hand radiograph for automated bone age estimation.")

def load_trained_model():
    model = ResNet50Baseline()
    checkpoint_path = "checkpoints/resnet50_baseline_best.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model checkpoint from {checkpoint_path}: {e}")
        return None

# Aspect-ratio preserving padding to 512x512
def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create black square canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

uploaded_file = st.file_uploader("Choose a hand X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    img_np = np.array(image)
    
    # Preprocessing with aspect-ratio preservation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    padded_resized = pad_and_resize(enhanced, target_size=512)
    rgb_img = cv2.cvtColor(padded_resized, cv2.COLOR_GRAY2RGB)
    
    # Side-by-side display using Streamlit columns (no stretching)
    col_raw, col_proc = st.columns(2)
    with col_raw:
        st.image(image, caption="Uploaded PA Radiograph", use_container_width=True)
    with col_proc:
        st.image(rgb_img, caption="CLAHE Preprocessed (512x512 Padded)", use_container_width=True)
    
    with st.spinner("Running model inference..."):
        model = load_trained_model()
        if model is not None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            tensor_img = transform(rgb_img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(tensor_img)
                predicted_age = output.item()
                
            st.markdown("---")
            st.success("### Diagnostic Assessment Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Bone Age", value=f"{predicted_age:.1f} Months")
            with col2:
                years = predicted_age / 12.0
                st.metric(label="Approximate Age in Years", value=f"{years:.1f} Years")
