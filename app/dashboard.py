# app/dashboard.py
import streamlit as st
from PIL import Image
import io
import requests
import numpy as np
import cv2
import json
import os
from io import BytesIO

# try to use local inference/gradcam modules if available
USE_LOCAL = True
try:
    from src.inference import predict_pil, _load
    from src.gradcam import make_gradcam_heatmap, overlay_heatmap
except Exception:
    USE_LOCAL = False

# Config
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI server
st.set_page_config(page_title="Plant Disease Detector", layout="wide", page_icon="🌿")

# Minimal CSS to improve look
# Minimal CSS to improve look
st.markdown(
    """
    <style>
    .big-title { 
        font-size: 34px; 
        font-weight: 700; 
        color: #145214;
    }
    .subtle { 
        color: #666; 
        font-size: 14px; 
    }
    .card { 
        background: linear-gradient(90deg, #ffffff, #f6fff6);
        padding: 14px;
        border-radius: 12px;
    }
    .pred { 
        font-size: 20px; 
        font-weight: 700; 
        color: #0b6b0b;
    }
    .small { 
        font-size: 13px; 
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.image("assets/logo.png" if os.path.exists("assets/logo.png") else None, width=80)

with col2:
    st.markdown('<div class="big-title">Plant Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload a leaf photo — the model will predict disease and show a Grad-CAM heatmap.</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar controls
with st.sidebar.image(os.path, width=150):
    st.header("Settings")
    api_mode = st.checkbox("Use FastAPI backend", value=True)
    if api_mode and not USE_LOCAL:
        st.info("Using API mode (FastAPI). Make sure uvicorn is running on port 8000.")
    st.write("Grad-CAM")
    enable_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)
    alpha = st.slider("Heatmap transparency", 0.0, 1.0, 0.45, 0.05)
    show_examples = st.checkbox("Show sample images", value=False)

    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Open API docs"):
        st.experimental_set_query_params(_open_docs="true")  # just a signal (users can open manually)

    st.markdown("---")
    st.markdown("Project")
    st.write("Model: EfficientNet (fine-tuned)")
    st.write("Saved model: `models/efficientnet_model.keras`")

# Show sample images
if show_examples:
    st.subheader("Sample images")
    sample_cols = st.columns(4)
    sample_dir = "assets/examples"
    if os.path.isdir(sample_dir):
        paths = sorted(os.listdir(sample_dir))[:4]
        for c, p in zip(sample_cols, paths):
            c.image(os.path.join(sample_dir, p), caption=p, use_column_width=True)
    else:
        st.info("Place sample images in `assets/examples/` to show them here.")

st.markdown("## Upload leaf image")
upload_col, info_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader("Upload JPG / PNG image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    # drag & drop hint
    st.markdown('<div class="small">Tip: drag & drop an image or use the sample images.</div>', unsafe_allow_html=True)
with info_col:
    st.markdown('<div class="card small">Info</div>', unsafe_allow_html=True)
    st.write("Model input size: 224×224")
    st.write("Confidence and probabilities shown")

if not uploaded_file:
    st.info("Upload an image to start prediction.")
    st.stop()

# read image
try:
    img = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Cannot open image: {e}")
    st.stop()

# show preview
st.markdown("### Preview")
preview_cols = st.columns([1, 1])
with preview_cols[0]:
    st.image(img, caption="Uploaded image", use_column_width=True)

# Send to API or run local inference
predict_btn = st.button("Predict", type="primary")
if not predict_btn:
    st.stop()

with st.spinner("Predicting..."):
    # 1) Try API mode if selected
    result = None
    api_error = None

    if api_mode:
        try:
            # ensure file bytes are rewound
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
            resp = requests.post(API_URL, files=files, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            api_error = str(e)
            result = None

    # 2) fallback to local inference if available and API failed or not selected
    if result is None and USE_LOCAL:
        try:
            res = predict_pil(img)  # predict_pil returns a dict
            result = res
        except Exception as e:
            api_error = api_error + " | local error: " + str(e) if api_error else str(e)
            result = None

    if result is None:
        st.error("Prediction failed. " + (api_error or "No backend available."))
        st.stop()

# result is expected: {"disease": "...", "confidence": 0.12, "probs": {...}}
disease = result.get("disease", "Unknown")
confidence = result.get("confidence", 0.0)
probs = result.get("probs", {})

# nice result card
st.markdown("### Result")
card_col1, card_col2 = st.columns([1, 1])
with card_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="pred">Predicted: {disease}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small">Confidence:</div>', unsafe_allow_html=True)
    st.metric(label="Confidence", value=f"{confidence:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

with card_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Top probabilities")
    # prepare top probs sorted
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:6]
    labels = [p[0] for p in top]
    scores = [p[1] for p in top]
    # simple horizontal bars using st.progress is limited, so draw a Matplotlib bar
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.barh(labels[::-1], scores[::-1], color="#66bb6a")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        # fallback plain text
        for lbl, s in top:
            st.write(f"- {lbl}: {s:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

# Grad-CAM: local model needed or compute locally with _load model
if enable_gradcam:
    st.markdown("### Grad-CAM")
    grad_col1, grad_col2 = st.columns(2)
    with grad_col1:
        st.write("Original")
        st.image(img, use_column_width=True)
    with grad_col2:
        st.write("Overlay")
        overlay_img = None
        grad_error = None
        if USE_LOCAL:
            try:
                # prepare model and array
                model, labels = _load()
                resized = img.resize((224, 224))
                arr = np.asarray(resized, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, 0)
                heatmap = make_gradcam_heatmap(arr, model)
                overlay = overlay_heatmap(np.array(img), heatmap, alpha=alpha)
                overlay_pil = Image.fromarray(overlay.astype("uint8"))
                st.image(overlay_pil, use_column_width=True)
            except Exception as e:
                grad_error = str(e)
                st.warning("Grad-CAM failed locally: " + grad_error)
        else:
            st.info("Grad-CAM requires the local model. Toggle off 'Use FastAPI backend' in sidebar to enable local Grad-CAM.")

# Download JSON result
st.markdown("---")
if st.button("Download prediction (JSON)"):
    out = {"disease": disease, "confidence": confidence, "probs": probs}
    b = BytesIO()
    b.write(json.dumps(out, indent=2).encode("utf-8"))
    b.seek(0)
    st.download_button("Download JSON", data=b, file_name="prediction.json", mime="application/json")

st.markdown("<div class='small'>Tip: For better Grad-CAM and local inference, run the dashboard with the same venv that has TensorFlow installed.</div>", unsafe_allow_html=True)
