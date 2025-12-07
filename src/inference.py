# src/inference.py
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Paths - adjust only if you changed them
KERAS_MODEL_PATH = "models/efficientnet_model.keras"      # .keras file we saved
LABELS_PATH = "src/labels.txt"
IMG_SIZE = (224, 224)

# lazy load
_model = None
_labels = None

def _load():
    global _model, _labels
    if _model is None:
        if not os.path.exists(KERAS_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {KERAS_MODEL_PATH}")
        _model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    if _labels is None:
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
        with open(LABELS_PATH, "r") as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _model, _labels

def predict_pil(img: Image.Image):
    """Take a PIL image, return dict {label, confidence, probs}"""
    model, labels = _load()
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return {
        "label": labels[idx],
        "confidence": float(preds[idx]),
        "probs": {labels[i]: float(preds[i]) for i in range(len(labels))}
    }

if __name__ == "__main__":
    # quick local test
    from PIL import Image
    img = Image.open("assets/sample_leaf.png")
    print(predict_pil(img))
