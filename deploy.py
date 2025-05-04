import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pydicom
import numpy as np
from skimage import exposure
from skimage.transform import resize
import os
import io

# Import your model architecture
from model import PatchCNN  # Make sure this points to your model definition

torch.classes.__path__ = []

# Set page title and header
st.title("NSCLC Classification Tool")
st.header("Upload CT scans to classify lung cancer subtypes")
st.write("This tool helps classify Non-Small Cell Lung Carcinoma (NSCLC) into subtypes: Adenocarcinoma (ADC) or Squamous Cell Carcinoma (SCC)")

# Create sidebar for model selection and options
st.sidebar.header("Model Options")
model_path = st.sidebar.selectbox(
    "Select model weights",
    ["best_patch_cnn.pth"]  # You can add more trained models here
)

# Function to load model
@st.cache_resource
def load_model(model_path):
    model = PatchCNN(num_classes=2)  # Assuming binary classification (ADC vs SCC)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
try:
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Create file uploader
upload_option = st.radio(
    "Select upload type:",
    ["DICOM file (.dcm)", "CT scan image (.jpg, .png)"]
)

if upload_option == "DICOM file (.dcm)":
    uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm"])
else:
    uploaded_file = st.file_uploader("Choose a CT scan image", type=["jpg", "jpeg", "png"])

# Process uploaded file
if uploaded_file is not None:
    # Display the uploaded file
    st.write("Processing your upload...")
    
    # Preprocess based on file type
    try:
        if upload_option == "DICOM file (.dcm)":
            # Read DICOM
            dicom_bytes = uploaded_file.read()
            dicom_buffer = io.BytesIO(dicom_bytes)
            ds = pydicom.dcmread(dicom_buffer)
            
            if 'PixelData' not in ds:
                st.error("This DICOM file does not contain image data.")
                st.stop()
                
            # Convert to image
            img = ds.pixel_array
            
            # Handle different image dimensions
            if len(img.shape) > 2:
                if len(img.shape) == 3 and img.shape[2] <= 4:  # RGB/RGBA
                    img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
                elif len(img.shape) == 3:  # 3D volume
                    img = img[img.shape[0]//2]  # Take middle slice
                    img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
                    img = np.stack([img] * 3, axis=-1)
                else:
                    st.error("Unsupported DICOM image dimensions")
                    st.stop()
            else:
                # 2D grayscale
                img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
                img = np.stack([img] * 3, axis=-1)
                
            # Display the image
            st.image(img, caption="Uploaded DICOM image", use_column_width=True)
            
        else:
            # Regular image file
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Display the image
            st.image(img, caption="Uploaded CT scan", use_column_width=True)
            img = img_array
            
        # Ensure image is properly sized
        if img.shape[0] < 400 or img.shape[1] < 400:
            # Resize to minimum dimensions
            new_size = (max(400, img.shape[0]), max(400, img.shape[1]), 3)
            img = resize(img, new_size, preserve_range=True).astype(np.uint8)
            
        # Extract multiple patches for prediction
        patches = []
        patch_size = 400
        
        # Extract up to 5 patches from different regions
        for i in range(min(5, (img.shape[0] - patch_size) // 100 + 1)):
            for j in range(min(5, (img.shape[1] - patch_size) // 100 + 1)):
                y = i * 100
                x = j * 100
                if y + patch_size <= img.shape[0] and x + patch_size <= img.shape[1]:
                    patch = img[y:y+patch_size, x:x+patch_size]
                    patches.append(patch)
        
        if not patches:
            # If no patches were created, use the whole image
            img = resize(img, (400, 400, 3), preserve_range=True).astype(np.uint8)
            patches = [img]
            
        # Transform patches for model input
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for patch in patches:
                patch_pil = Image.fromarray(patch)
                patch_tensor = transform(patch_pil).unsqueeze(0)
                output = model(patch_tensor)
                prob = torch.softmax(output, dim=1).squeeze().numpy()
                predictions.append(prob)
                
        # Average predictions across patches
        avg_prediction = np.mean(predictions, axis=0)
        
        # Display results
        st.write("## Classification Results")
        
        classes = ["Adenocarcinoma (ADC)", "Squamous Cell Carcinoma (SCC)"]
        predicted_class = np.argmax(avg_prediction)
        
        st.write(f"**Predicted class:** {classes[predicted_class]}")
        st.write(f"**Confidence:** {avg_prediction[predicted_class]*100:.2f}%")
        
        # Display probability bar chart
        st.write("### Probability Distribution")
        st.bar_chart({classes[i]: avg_prediction[i] for i in range(len(classes))})
        
        # Add disclaimer
        st.write("---")
        st.write("**Disclaimer:** This tool is for research purposes only and should not be used for clinical diagnosis without proper validation.")
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
