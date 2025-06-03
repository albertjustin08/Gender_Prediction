import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pred_gender_model_v1.keras")

model = load_model()

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Gender detection transformer
class GenderDetectionTransformer(VideoTransformerBase):
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

                face_crop = img[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue

                face_resized = cv2.resize(face_crop, (224, 224))
                face_array = np.expand_dims(face_resized, axis=0)
                prediction = model.predict(face_array, verbose=0)
                label = "Female" if np.argmax(prediction, axis=1)[0] == 0 else "Male"

                color = (0, 255, 0) if label == "Male" else (255, 0, 255)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return img

# Streamlit UI
st.title("📸 Real-Time Gender Detection")
st.markdown("Using webcam and Mediapipe + TensorFlow for live face-based gender classification.")

webrtc_streamer(
    key="gender-detection",
    video_transformer_factory=GenderDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
