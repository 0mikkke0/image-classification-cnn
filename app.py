from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cifar10_cnn.keras"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️")
st.title("🖼️ CIFAR-10 Image Classifier")
st.caption("Deep-learning portfolio project using a Convolutional Neural Network.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Classify Image", type="primary"):
        resized = image.resize((32, 32))
        array = np.asarray(resized, dtype=np.float32)[None, ...]
        probabilities = model.predict(array, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        st.subheader("Prediction")
        best = top_indices[0]
        st.success(
            f"{CLASS_NAMES[best].title()} — {probabilities[best]:.1%} confidence"
        )

        st.subheader("Top 3 predictions")
        for idx in top_indices:
            st.write(
                f"**{CLASS_NAMES[idx].title()}** — {probabilities[idx]:.1%}"
            )
