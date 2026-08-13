from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# -----------------------------
# Configuration
# -----------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cifar10_cnn.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ CIFAR-10 Image Classifier")
st.caption(
    "Deep-learning portfolio project using a Convolutional Neural Network."
)


# -----------------------------
# Load model
# -----------------------------

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Please make sure "
        "`models/cifar10_cnn.keras` exists."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# -----------------------------
# Image upload
# -----------------------------

st.write("Upload an image and let the CNN classify it.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=350,
    )

    if st.button("🔍 Classify Image", type="primary"):

        # CIFAR-10 images are 32x32
        resized = image.resize((32, 32))

        array = np.asarray(
            resized,
            dtype=np.float32,
        )[None, ...]

        probabilities = model.predict(
            array,
            verbose=0,
        )[0]

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        best = top_indices[0]

        # Main prediction
        st.subheader("Prediction")

        st.success(
            f"{CLASS_NAMES[best].title()} — "
            f"{probabilities[best]:.1%} confidence"
        )

        # Top 3 predictions
        st.subheader("Top 3 Predictions")

        for idx in top_indices:
            st.write(
                f"**{CLASS_NAMES[idx].title()}** — "
                f"{probabilities[idx]:.1%}"
            )

else:
    st.info(
        "Upload a JPG, JPEG, or PNG image to get started."
    )


# -----------------------------
# Model information
# -----------------------------

with st.expander("About this model"):

    st.write(
        """
        This application uses a Convolutional Neural Network (CNN)
        trained on the CIFAR-10 dataset.

        The model can classify images into 10 categories:
        """
    )

    st.write(
        ", ".join(
            name.title()
            for name in CLASS_NAMES
        )
    )

    st.write("Test accuracy: **74.09%**")