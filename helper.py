from ultralytics import YOLO
import streamlit as st

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def detect_objects_image(image, model, confidence):
    """
    Detects objects in an image using the YOLOv8 model.

    Parameters:
        image (PIL.Image): The image uploaded by the user.
        model (YOLO): The YOLOv8 object detection model.
        confidence (float): Confidence threshold for object detection.

    Returns:
        None
    """
    res = model.predict(image, conf=confidence)
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Image',
             use_column_width=True)
