import streamlit as st
import tensorflow as tf
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import cv2
import requests
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model
model = tf.keras.models.load_model("pred_gender_model_v1.keras")

# Setup Mediapipe
face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6,
    static_image_mode=True  # Hindari GPU/GL context
)


st.set_page_config(page_title="Gender Detection", layout="centered")
st.title("👁️ Gender Detection - Webcam | Upload | URL")

# Fungsi prediksi wajah
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

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📸 Live Webcam", "🖼️ Upload Image", "🌐 Image from URL"])

# --- Tab 1: Webcam Realtime ---
with tab1:
    st.markdown("### Realtime Gender Detection via Webcam")

    class GenderVideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, _ = img_rgb.shape
            results = face_detector.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * img_width)
                    y_min = int(bbox.ymin * img_height)
                    w = int(bbox.width * img_width)
                    h = int(bbox.height * img_height)
                    x_max = x_min + w
                    y_max = y_min + h

                    face = img_rgb[y_min:y_max, x_min:x_max]
                    if face.size == 0:
                        continue

                    resized = cv2.resize(face, (224, 224))
                    input_array = np.expand_dims(resized, axis=0)
                    prediction = model.predict(input_array)
                    label = "Female" if np.argmax(prediction) == 0 else "Male"

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            return img

    webrtc_streamer(key="gender-webcam", video_transformer_factory=GenderVideoProcessor)

# --- Tab 2: Upload Image ---
with tab2:
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        label, result_img = predict_gender(image)
        st.image(result_img, caption=f"Prediction: {label}", use_column_width=True)

# --- Tab 3: Image from URL ---
with tab3:
    st.markdown("### Enter Image URL")
    url = st.text_input("Image URL")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            label, result_img = predict_gender(image)
            st.image(result_img, caption=f"Prediction: {label}", use_column_width=True)
        except:
            st.error("Failed to load image from URL.")
