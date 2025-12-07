# src/inference.py
"""
Robust inference helper used by app/api.py and app/dashboard.py.

Provides:
- _load(): lazy load model + labels
- predict_pil(PIL.Image) -> {"label":..., "confidence":..., "probs": {...}}
- predict_bytes(bytes) -> convenience wrapper (not required)
"""

import os
import threading
from typing import Dict, Any
from PIL import Image
import numpy as np
import tensorflow as tf

# --------- Config (edit only if your paths are different) ----------
KERAS_MODEL_PATH = "models/efficientnet_model.keras"   # .keras model path
LABELS_PATH = "src/labels.txt"                         # one label per line, same order as training
IMG_SIZE = (224, 224)
# -------------------------------------------------------------------

# robust import of preprocess_input across TF versions
try:
    from tensorflow.keras.applications import efficientnet
    preprocess_input = efficientnet.preprocess_input
except Exception:
    try:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except Exception:
        preprocess_input = None  # fallback to /255.0 if not available

# lazy-loaded objects and lock for thread safety
_model = None
_labels = None
_lock = threading.Lock()

def _load():
    """Lazy load the keras model and labels. Returns (model, labels)."""
    global _model, _labels
    with _lock:
        if _model is None:
            if not os.path.exists(KERAS_MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {KERAS_MODEL_PATH}")
            _model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        if _labels is None:
            if not os.path.exists(LABELS_PATH):
                raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _labels = [line.strip() for line in f if line.strip()]
    return _model, _labels

def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image -> model-ready numpy array (1, H, W, 3)
    Uses EfficientNet's preprocess_input when available; otherwise scales 0..1.
    """
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")
    if preprocess_input is not None:
        arr = preprocess_input(arr)
    else:
        # safe fallback
        arr = arr / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def predict_pil(img: Image.Image) -> Dict[str, Any]:
    """
    Predict from a PIL Image.
    Returns dict: { "label": str, "confidence": float, "probs": {label:prob, ...} }
    """
    model, labels = _load()
    x = _preprocess_pil(img)
    preds = model.predict(x, verbose=0)

    # preds may be shape (1, N) or (N,) etc.
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        raw = preds[0]
    elif preds.ndim == 1:
        raw = preds
    else:
        raw = preds.ravel()

    # If outputs are already probabilities (sum ~1), softmax is harmless; apply for safety.
    try:
        probs = tf.nn.softmax(raw).numpy()
    except Exception:
        # if tf not available for some reason, use numpy
        exps = np.exp(raw - np.max(raw))
        probs = exps / np.sum(exps)

    # guard: if label count mismatch, trim/pad
    n_model = probs.shape[0]
    if len(labels) != n_model:
        # Warning: labels length differs from model output
        # Try to align by cropping or padding labels with placeholders
        if len(labels) > n_model:
            labels = labels[:n_model]
        else:
            labels = labels + [f"CLASS_{i}" for i in range(n_model - len(labels))]

    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_conf = float(probs[top_idx])

    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return {"label": top_label, "confidence": top_conf, "probs": probs_dict}

def predict_bytes(b: bytes) -> Dict[str, Any]:
    """Convenience wrapper if you have image bytes (e.g. from FastAPI UploadFile)."""
    img = Image.open(io.BytesIO(b))
    return predict_pil(img)
