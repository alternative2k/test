"""
Streamlit Webcam Person Detection App

This app uses Ultralytics YOLO and its Streamlit live inference helper
to run real-time person detection on your webcam.
"""

import streamlit as st
from ultralytics import solutions

def main():
    st.set_page_config(page_title="Webcam Person Detection", layout="wide")
    st.title("Webcam Person Detection with YOLO")
    st.write(
        "This app runs real-time object detection on your webcam feed. "
        "It uses a YOLO model and displays bounding boxes around detected people."
    )

    # Sidebar configuration
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "YOLO model",
        options=["yolo11n.pt", "yolo11s.pt"],
        index=0,
        help="Smaller models (n) are faster; larger ones are more accurate."
    )
    conf = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Minimum confidence for a detection to be shown."
    )
    iou = st.sidebar.slider(
        "NMS IoU threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Intersection over Union threshold for non-max suppression."
    )

    st.sidebar.info(
        "Click **Start** in the widget below to begin webcam inference. "
        "Allow camera access when prompted in your browser."
    )

    # Create the live inference solution
    # The solutions.Inference helper handles webcam/video capture and drawing.
    inf = solutions.Inference(
        model=model_name,
        conf=conf,
        iou=iou,
        source=0,        # 0 = default webcam
        show=False,      # Let Streamlit handle display
        classes=[0],     # 0 is 'person' class in COCO
        streamlit=True,  # Enable Streamlit UI
    )

    # Launch the Streamlit-integrated inference UI
    inf.inference()


if __name__ == "__main__":
    main()
    # Run with: streamlit run app.py
