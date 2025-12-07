import sys, os
# add project root to path so local package imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import _load
from src.gradcam import make_gradcam_heatmap, overlay_heatmap

from PIL import Image
import numpy as np
import cv2

model, labels = _load()  # loads your saved .keras model
print("Model loaded, last layers:", [l.name for l in model.layers[-10:]])

# open image
img = Image.open("assets/sample_leaf.png").convert("RGB")
img_resized = img.resize((224, 224))
arr = np.array(img_resized, dtype=np.float32) / 255.0
arr = np.expand_dims(arr, 0)

# make heatmap
heatmap = make_gradcam_heatmap(arr, model)  # tries to infer conv layer
print("Heatmap shape:", heatmap.shape, "max:", heatmap.max())

# overlay on original full-size image
orig = np.array(img)
overlay = overlay_heatmap(orig, heatmap)
cv2.imwrite("assets/gradcam_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("Saved overlay to assets/gradcam_overlay.png")
