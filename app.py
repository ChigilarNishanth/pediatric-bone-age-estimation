import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from src.baseline_resnet50 import ResNet50Baseline

st.set_page_config(page_title="Pediatric Bone Age Estimation", layout="centered")

st.title("Pediatric Bone Age & Sex Estimation")
st.write("Upload a postero-anterior (PA) hand radiograph for automated bone age estimation.")

uploaded_file = st.file_uploader("Choose a hand X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Radiograph", use_column_width=True)
    
    # Preprocess with CLAHE
    img_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    resized = cv2.resize(enhanced, (512, 512))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    st.subheader("CLAHE Preprocessed Output (512x512)")
    st.image(rgb_img, caption="OpenCV CLAHE Enhanced Image", use_column_width=True)
