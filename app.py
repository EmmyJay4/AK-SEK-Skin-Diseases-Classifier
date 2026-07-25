import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pathlib

st.set_page_config(page_title="AK vs SEK Skin Lesion Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/best_model.keras")
    return model

def predict(model, pil_image):
    class_names = ["AK", "SEK"]
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = class_names[pred_idx]
    ak_pct = float(probs[0]) * 100
    sek_pct = float(probs[1]) * 100
    return label, ak_pct, sek_pct

st.title("AK vs SEK Skin Lesion Classifier")
st.write("Upload a skin lesion image to classify it as **Actinic Keratosis (AK)** or **Seborrheic Keratoses (SEK)**.")

model = load_model()
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    label, ak_pct, sek_pct = predict(model, img)

    st.write(f"**Prediction:** {label}")
    st.progress(int(ak_pct), text=f"AK: {ak_pct:.1f}%")
    st.progress(int(sek_pct), text=f"SEK: {sek_pct:.1f}%")
