"""
DeepFER Streamlit Dashboard
============================
Run with:  streamlit run app.py

Pages:
  - Live Prediction : upload a photo or use your webcam, get an emotion
                       prediction with confidence breakdown and Grad-CAM
                       explainability overlay
  - Model Insights   : training curves, confusion matrix, per-class metrics
  - Dataset Explorer : class distribution + sample images (reads your
                       local dataset directly, no retraining needed)
  - About            : project overview
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.face_utils import detect_faces, crop_and_prepare_face, prepare_whole_image, FACE_DETECTION_AVAILABLE

import tensorflow as tf

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DeepFER — Facial Emotion Recognition", page_icon="🙂", layout="wide")

CLASS_COLORS = {
    "angry": "#e63946", "disgust": "#6a994e", "fear": "#5a189a", "happy": "#f4a261",
    "neutral": "#8d99ae", "sad": "#457b9d", "surprise": "#e9c46a",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(config.MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.MODEL_PATH)


@st.cache_data
def load_label_map():
    if not os.path.exists(config.LABEL_MAP_PATH):
        return {str(i): name for i, name in enumerate(config.CLASS_NAMES)}
    with open(config.LABEL_MAP_PATH) as f:
        return json.load(f)


@st.cache_data
def load_history():
    if not os.path.exists(config.HISTORY_PATH):
        return None
    with open(config.HISTORY_PATH) as f:
        return json.load(f)


@st.cache_data
def load_eval_report():
    if not os.path.exists(config.EVAL_REPORT_PATH):
        return None
    with open(config.EVAL_REPORT_PATH) as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix():
    if not os.path.exists(config.CONFUSION_MATRIX_PATH):
        return None
    return np.load(config.CONFUSION_MATRIX_PATH)


@st.cache_data
def count_dataset_images():
    counts = {"train": {}, "test": {}}
    for split, directory in [("train", config.TRAIN_DIR), ("test", config.TEST_DIR)]:
        if not os.path.isdir(directory):
            continue
        for cname in config.CLASS_NAMES:
            class_dir = os.path.join(directory, cname)
            if os.path.isdir(class_dir):
                counts[split][cname] = len(os.listdir(class_dir))
    return counts


model = load_model()
label_map = load_label_map()
idx_to_class = {int(k): v for k, v in label_map.items()}


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_face(face_array: np.ndarray):
    """face_array: (48, 48) float32 in [0, 1]. Returns (probs, pred_idx, heatmap_overlay)."""
    input_tensor = np.expand_dims(np.expand_dims(face_array, 0), -1)
    probs = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    heatmap, _, _ = make_gradcam_heatmap(input_tensor, model)
    overlay = overlay_heatmap(face_array, heatmap)
    return probs, pred_idx, overlay


def render_prediction_result(face_array: np.ndarray, source_label: str = ""):
    probs, pred_idx, overlay = predict_face(face_array)
    pred_class = idx_to_class[pred_idx]
    emoji = config.EMOJI.get(pred_class, "")

    col1, col2, col3 = st.columns([1, 1, 1.4])

    with col1:
        st.markdown(f"**Input {source_label}**")
        st.image((face_array * 255).astype("uint8"), width=180, clamp=True)

    with col2:
        st.markdown("**Grad-CAM — what the model focused on**")
        st.image(overlay, width=180)

    with col3:
        st.markdown(f"### {emoji} Prediction: **{pred_class.capitalize()}**")
        st.markdown(f"Confidence: **{probs[pred_idx]:.1%}**")

        prob_df = pd.DataFrame({
            "emotion": [idx_to_class[i].capitalize() for i in range(len(probs))],
            "probability": probs,
        }).sort_values("probability", ascending=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = [CLASS_COLORS.get(idx_to_class[i], "#457b9d") for i in prob_df.index]
        ax.barh(prob_df["emotion"], prob_df["probability"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")
        for i, v in enumerate(prob_df["probability"]):
            ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=8)
        st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🙂 DeepFER")
page = st.sidebar.radio("Navigate", ["Live Prediction", "Model Insights", "Dataset Explorer", "About"])

if model is None:
    st.sidebar.warning(
        "No trained model found at `models/deepfer_model.keras`. "
        "Run `python src/train.py` first, then refresh this app."
    )


# ---------------------------------------------------------------------------
# PAGE: Live Prediction
# ---------------------------------------------------------------------------
if page == "Live Prediction":
    st.title("Live Emotion Prediction")
    st.caption("Upload a photo or use your webcam. Faces are detected automatically, cropped, and classified.")

    if model is None:
        st.error("Train the model first (`python src/train.py`) — this page needs a saved model to run predictions.")
        st.stop()

    input_mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)

    pil_image = None
    if input_mode == "Upload Image":
        uploaded = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
    else:
        cam_photo = st.camera_input("Take a photo")
        if cam_photo is not None:
            pil_image = Image.open(cam_photo).convert("RGB")

    if not FACE_DETECTION_AVAILABLE:
        st.info(
            "Face auto-detection isn't available in this OpenCV install (no usable Haar cascade "
            "file was found). Predictions will run on the whole uploaded image instead — this "
            "works best if the photo is already a cropped, front-facing face. "
            "See the README for how to fix face detection."
        )

    if pil_image is not None:
        st.divider()
        faces, gray_array = detect_faces(pil_image)

        if len(faces) == 0:
            if FACE_DETECTION_AVAILABLE:
                st.warning("No face detected — running prediction on the full image instead.")
            face_arr = prepare_whole_image(pil_image)
            render_prediction_result(face_arr, source_label="(full image)")
        else:
            st.success(f"Detected {len(faces)} face(s).")
            # Show original with bounding boxes
            import cv2
            display_img = np.array(pil_image).copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (67, 133, 244), 3)
            st.image(display_img, caption="Detected face(s)", width=400)

            for i, box in enumerate(faces):
                st.divider()
                st.markdown(f"#### Face {i + 1}")
                face_arr = crop_and_prepare_face(gray_array, box)
                render_prediction_result(face_arr, source_label=f"— Face {i + 1}")


# ---------------------------------------------------------------------------
# PAGE: Model Insights
# ---------------------------------------------------------------------------
elif page == "Model Insights":
    st.title("Model Insights")
    st.caption("Training curves and evaluation metrics, generated by `src/train.py` and `src/evaluate.py`.")

    history = load_history()
    report = load_eval_report()
    cm = load_confusion_matrix()

    if history is None and report is None:
        st.info(
            "No training history or evaluation report found yet. "
            "Run `python src/train.py` then `python src/evaluate.py` to populate this page."
        )
    else:
        if history is not None:
            st.subheader("Training Curves")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.plot(history["accuracy"], label="Train")
                ax.plot(history["val_accuracy"], label="Validation")
                ax.set_title("Accuracy over Epochs"); ax.set_xlabel("Epoch"); ax.legend()
                st.pyplot(fig, use_container_width=True)
            with col2:
                fig, ax = plt.subplots()
                ax.plot(history["loss"], label="Train")
                ax.plot(history["val_loss"], label="Validation")
                ax.set_title("Loss over Epochs"); ax.set_xlabel("Epoch"); ax.legend()
                st.pyplot(fig, use_container_width=True)

        if report is not None:
            st.subheader("Per-Class Performance")
            rows = []
            for cname in config.CLASS_NAMES:
                if cname in report:
                    rows.append({
                        "Class": cname.capitalize(),
                        "Precision": report[cname]["precision"],
                        "Recall": report[cname]["recall"],
                        "F1-score": report[cname]["f1-score"],
                        "Support": int(report[cname]["support"]),
                    })
            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}"}),
                         use_container_width=True)
            st.metric("Overall Accuracy", f"{report['accuracy']:.2%}")

            fig, ax = plt.subplots(figsize=(9, 4))
            metrics_df.set_index("Class")[["Precision", "Recall", "F1-score"]].plot.bar(ax=ax)
            ax.set_ylim(0, 1); ax.set_title("Per-Class Precision / Recall / F1")
            st.pyplot(fig, use_container_width=True)

        if cm is not None:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[c.capitalize() for c in config.CLASS_NAMES],
                        yticklabels=[c.capitalize() for c in config.CLASS_NAMES], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Dataset Explorer
# ---------------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.caption("Reads directly from your local `archive-3/archive-3/train` and `test` folders.")

    counts = count_dataset_images()
    if not counts["train"] and not counts["test"]:
        st.warning(f"No dataset found at `{config.TRAIN_DIR}`. Update `BASE_DIR` in config.py.")
    else:
        col1, col2 = st.columns(2)
        for col, split in zip([col1, col2], ["train", "test"]):
            with col:
                st.subheader(f"{split.capitalize()} Set")
                split_df = pd.DataFrame(list(counts[split].items()), columns=["class", "count"])
                fig, ax = plt.subplots()
                colors = [CLASS_COLORS.get(c, "#457b9d") for c in split_df["class"]]
                ax.bar(split_df["class"], split_df["count"], color=colors)
                ax.set_xticklabels(split_df["class"], rotation=45)
                st.pyplot(fig, use_container_width=True)
                st.dataframe(split_df, use_container_width=True)

        st.subheader("Sample Images per Class")
        cols = st.columns(len(config.CLASS_NAMES))
        for col, cname in zip(cols, config.CLASS_NAMES):
            class_dir = os.path.join(config.TRAIN_DIR, cname)
            with col:
                st.markdown(f"**{cname.capitalize()}**")
                if os.path.isdir(class_dir):
                    files = os.listdir(class_dir)[:1]
                    if files:
                        st.image(os.path.join(class_dir, files[0]), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
else:
    st.title("About DeepFER")
    st.markdown("""
DeepFER is a facial emotion recognition system that classifies faces into 7 emotions —
**angry, disgust, fear, happy, neutral, sad, surprise** — using a custom CNN trained on the
FER2013 dataset (48x48 grayscale images).

**How it works:**
1. A face is detected in the uploaded photo or webcam frame using OpenCV.
2. The face is cropped, resized to 48x48, and normalized.
3. A CNN (3 convolutional blocks with batch normalization, dropout, and class-weighted
   training to handle dataset imbalance) predicts a probability for each emotion.
4. Grad-CAM highlights which facial regions most influenced the prediction, so the result
   isn't just a label — you can see *why* the model made that call.

**Why class weighting matters:** the training data is imbalanced — the 'happy' class has
roughly 16x more images than 'disgust'. Without correcting for this, a model can reach
deceptively high overall accuracy while almost never correctly identifying rarer emotions.
""")
