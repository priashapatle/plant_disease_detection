# src/gradcam.py
import tensorflow as tf
import numpy as np
import cv2

def _infer_last_conv_layer_name(model):
    # Heuristic: look for a layer name containing "top_conv" or the last layer with "conv"
    names = [layer.name for layer in model.layers]
    for name in reversed(names):
        if "top_conv" in name:
            return name
    for name in reversed(names):
        if "conv" in name:
            return name
    # fallback: return None so caller must handle it
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    img_array: numpy array shape (1, H, W, 3), normalized to 0..1
    model: keras model
    last_conv_layer_name: optional string; if None we try to infer one
    returns: heatmap float array HxW with values in 0..1
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _infer_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        raise ValueError("Could not infer last conv layer name. Pass last_conv_layer_name explicitly.")

    # Build a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # compute gradients of the top predicted class w.r.t. conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # vector where each entry is the mean intensity of the gradient over a feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    # weight the conv outputs with the pooled gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # average over channels to get the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)

    # Relu and normalize to 0..1
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-8
    heatmap /= max_val
    return heatmap

def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    """
    img_rgb: uint8 HxW x3 (RGB)
    heatmap: float HxW 0..1
    alpha: float blend factor
    Returns overlay image uint8 HxW x3
    """
    if heatmap is None:
        raise ValueError("heatmap is None")

    # resize heatmap to image size and convert to color map
    hmap = cv2.resize((heatmap * 255).astype("uint8"), (img_rgb.shape[1], img_rgb.shape[0]))
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    # make sure img is uint8 BGR for cv2 addWeighted: convert RGB->BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, hmap_color, alpha, 0)

    # convert back to RGB before returning
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb
