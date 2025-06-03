import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import mediapipe as mp

# Load the model
model = tf.keras.models.load_model("pred_gender_model_v1.keras")

# Initialize Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Function to predict gender
def predict_gender(pil_img):
    image_np = np.array(pil_img.convert("RGB"))
    img_height, img_width, _ = image_np.shape
    results = face_detector.process(image_np)

    if not results.detections:
        return "No face detected", pil_img

    draw = ImageDraw.Draw(pil_img)
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    x_min = int(bbox.xmin * img_width)
    y_min = int(bbox.ymin * img_height)
    box_width = int(bbox.width * img_width)
    box_height = int(bbox.height * img_height)
    x_max = x_min + box_width
    y_max = y_min + box_height

    face_crop = image_np[y_min:y_max, x_min:x_max]
    if face_crop.size == 0:
        return "Invalid face crop", pil_img

    face_resized = cv2.resize(face_crop, (224, 224))
    face_array = np.expand_dims(face_resized, axis=0)
    prediction = model.predict(face_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = "Female" if predicted_class == 0 else "Male"

    draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
    draw.text((x_min, y_min - 10), label, fill="red")
    return label, pil_img

# Streamlit UI
st.title("👩‍🦰 Gender Detection App")
st.markdown("Upload an image or provide a URL to detect gender based on face.")

# Image inputs
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL")

# Webcam snapshot (optional - desktop browser only)
use_camera = st.checkbox("Use webcam (experimental)")
if use_camera:
    camera_img = st.camera_input("Take a picture")

# Predict button
if st.button("Predict Gender"):
    try:
        pil_img = None

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
        elif image_url:
            response = requests.get(image_url)
            pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        elif use_camera and camera_img is not None:
            pil_img = Image.open(camera_img).convert("RGB")
        else:
            st.warning("Please provide an image by upload, URL, or webcam.")
        
        if pil_img:
            label, output_img = predict_gender(pil_img)
            st.success(f"Prediction: **{label}**")
            st.image(output_img, caption=f"Detected Gender: {label}", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
