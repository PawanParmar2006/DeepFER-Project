"""
Central configuration for DeepFER.

Every module — training, evaluation, and the Streamlit app — imports from
here instead of hardcoding paths or hyperparameters, so training and
inference always agree on image size, class order, and preprocessing.
"""

import os

# ---------------------------------------------------------------------------
# Paths — update BASE_DIR to wherever your dataset actually lives
# ---------------------------------------------------------------------------
BASE_DIR = r"C:/Users/pawan/Downloads/FER/deepfer/archive-3/archive-3"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfer_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
EVAL_REPORT_PATH = os.path.join(MODEL_DIR, "eval_report.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.npy")

# ---------------------------------------------------------------------------
# Image / data parameters
# ---------------------------------------------------------------------------
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_CHANNELS = 1
COLOR_MODE = "grayscale"

BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
SEED = 42

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)

EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4

# Name of the last conv layer in src/model.py — used by Grad-CAM
LAST_CONV_LAYER = "last_conv"
