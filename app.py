# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# ML Model Config
model_type = st.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Input for image upload
uploaded_image = st.file_uploader("Upload an image...",
                                   type=("jpg", "jpeg", "png", "bmp", "webp"))
if uploaded_image is not None:
    # Resize uploaded image to a maximum size of 800x800 while preserving aspect ratio
    max_size = (800, 800)
    image = Image.open(uploaded_image)
    image.thumbnail(max_size, Image.LANCZOS)  # Resize while preserving aspect ratio
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button('Detect Objects'):
        try:
            helper.detect_objects_image(image, model, confidence)
        except Exception as ex:
            st.error("Error occurred while processing the image.")
            st.error(ex)
