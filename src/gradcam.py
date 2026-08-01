"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for DeepFER.

Produces a heatmap over the input face showing which regions most
influenced the model's prediction — used in the Streamlit app's
"Why this prediction?" panel.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str = None):
    """img_array: shape (1, H, W, 1), already normalized to [0, 1]."""
    last_conv_layer_name = last_conv_layer_name or config.LAST_CONV_LAYER
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """original_img: 2D array [0,1] or [0,255], grayscale. Returns a PIL RGB image with heatmap overlay."""
    h, w = original_img.shape[:2]
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype("uint8")).resize((w, h)))
    jet = cm.jet(heatmap_resized / 255.0)[..., :3]

    base = original_img.astype("float32")
    if base.max() <= 1.0:
        base = base * 255.0
    base_rgb = np.stack([base] * 3, axis=-1) / 255.0

    blended = (1 - alpha) * base_rgb + alpha * jet
    blended = np.clip(blended * 255, 0, 255).astype("uint8")
    return Image.fromarray(blended)
