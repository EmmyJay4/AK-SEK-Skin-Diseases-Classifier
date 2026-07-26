import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pathlib

st.set_page_config(page_title="AK vs SEK Skin Lesion Classifier", page_icon="🧬", layout="centered")

st.markdown("""
<style>
    .app-header {
        border-bottom: 3px solid #1e3a5f;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 900;
        color: #1e3a5f;
        margin-bottom: 0.1rem;
    }
    .app-header .tag {
        color: #64748b;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .team-strip {
        display: flex;
        gap: 1.5rem;
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .disclaimer {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        color: #475569;
        margin-bottom: 1.5rem;
        border-left: 3px solid #94a3b8;
    }
    .upload-label {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .verdict {
        border-radius: 10px;
        padding: 1.4rem;
        margin-top: 1.2rem;
    }
    .verdict-ak {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .verdict-sek {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .verdict-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 0.2rem;
    }
    .verdict-name {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1e293b;
    }
    .footer-line {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-top: 2.5rem;
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

st.markdown("""
<div class="app-header">
    <h1>🧬 AK vs SEK Skin Lesion Classifier</h1>
    <div class="tag">Actinic Keratosis vs Seborrheic Keratoses</div>
    <div class="team-strip">
        <span>📚 GET 324 Mini-Project</span>
        <span>🏫 Department of Civil Engineering, University of Uyo</span>
        <span>👥 Group CV14</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚕️ This tool distinguishes between '
    '<b>Actinic Keratosis (AK)</b> and <b>Seborrheic Keratoses (SEK)</b> for academic '
    'demonstration purposes. It is not a substitute for professional medical diagnosis.</div>',
    unsafe_allow_html=True
)

st.markdown('<p class="upload-label">Upload a skin lesion image</p>', unsafe_allow_html=True)
model = load_model()
uploaded_file = st.file_uploader(" ", label_visibility="collapsed")

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't open this file as an image. Please try a JPG, PNG, WEBP, BMP, or similar image format.")
        st.stop()

    st.image(img, use_container_width=True)

    with st.spinner("Analyzing lesion pattern..."):
        label, ak_pct, sek_pct = predict(model, img)

    verdict_class = "verdict-ak" if label == "AK" else "verdict-sek"
    full_name = "Actinic Keratosis" if label == "AK" else "Seborrheic Keratoses"

    st.markdown(f"""
    <div class="verdict {verdict_class}">
        <div class="verdict-title">Predicted Condition</div>
        <div class="verdict-name">{full_name} ({label})</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AK likelihood", f"{ak_pct:.1f}%")
    with col2:
        st.metric("SEK likelihood", f"{sek_pct:.1f}%")

st.markdown(
    '<div class="footer-line">MobileNetV3Small Transfer Learning · 93.8% Test Accuracy · '
    'Trained on FYP Skin Disease Dataset (Kaggle)</div>',
    unsafe_allow_html=True
    )
