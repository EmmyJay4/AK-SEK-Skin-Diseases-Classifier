import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pathlib

st.set_page_config(page_title="AK vs SEK Skin Lesion Classifier", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .ak-box {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .sek-box {
        background-color: #dcfce7;
        color: #166534;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

st.markdown('<div class="main-title">🩺 AK vs SEK Skin Lesion Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a skin lesion image to classify it as '
    '<b>Actinic Keratosis (AK)</b> or <b>Seborrheic Keratoses (SEK)</b></div>',
    unsafe_allow_html=True
)

model = load_model()
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            label, ak_pct, sek_pct = predict(model, img)

        box_class = "ak-box" if label == "AK" else "sek-box"
        full_name = "Actinic Keratosis" if label == "AK" else "Seborrheic Keratoses"
        st.markdown(
            f'<div class="result-box {box_class}">Prediction: {label}<br>'
            f'<span style="font-size:0.9rem; font-weight:400;">{full_name}</span></div>',
            unsafe_allow_html=True
        )

        st.write("")
        st.progress(int(ak_pct), text=f"AK: {ak_pct:.1f}%")
        st.progress(int(sek_pct), text=f"SEK: {sek_pct:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit • MobileNetV3Small Transfer Learning Model • GET 324 Group CV14")
