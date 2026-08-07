from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cifar10_cnn.keras"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def predict_image(image: Image.Image):
    model = tf.keras.models.load_model(MODEL_PATH)
    image = image.convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index])
