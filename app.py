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

# Model Loading Function
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

# File Upload
uploaded_file = st.file_uploader("Choose a hand X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Display Uploaded Radiograph (Fixed deprecated parameter)
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded PA Radiograph", use_container_width=True)
    
    # 2. OpenCV CLAHE Preprocessing Pipeline (clipLimit=2.0, tileGridSize=(8,8))
    img_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    resized = cv2.resize(enhanced, (512, 512))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    st.subheader("CLAHE Preprocessed Output (512x512)")
    st.image(rgb_img, caption="OpenCV CLAHE Enhanced Image", use_container_width=True)
    
    # 3. Perform Deep Learning Inference
    with st.spinner("Running model inference..."):
        model = load_trained_model()
        if model is not None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            tensor_img = transform(rgb_img).unsqueeze(0)  # Shape: (1, 3, 512, 512)
            
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
