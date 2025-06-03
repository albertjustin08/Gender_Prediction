import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load model
model = tf.keras.models.load_model("pred_gender_model_v1 (1).keras")

# Fungsi prediksi
def predict(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    label = "Female" if predicted_class == 0 else "Male"
    return label

# UI Streamlit
st.set_page_config(layout="centered")
st.title("Gender Prediction (Simple Image Upload / URL)")
tab1, tab2 = st.tabs(["📁 Upload", "🌐 URL"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image of a face", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label = predict(image)
        st.success(f"Prediction: **{label}**")

with tab2:
    url = st.text_input("Paste Image URL")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)
            label = predict(image)
            st.success(f"Prediction: **{label}**")
        except Exception as e:
            st.error("Failed to load image from URL.")
