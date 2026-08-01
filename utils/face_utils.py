"""
Face detection + preprocessing helpers for the Streamlit app.

The FER2013 training images are already pre-cropped 48x48 grayscale faces.
Real photos a user uploads (selfies, group photos) are not — this module
detects and crops face regions first, using OpenCV's built-in Haar cascade,
so the app degrades gracefully from "toy demo" to "works on an actual photo."

Different OpenCV builds ship the haarcascade XML data files in different
locations (this shifted again in OpenCV 5's objdetect/xobjdetect split), so
rather than hardcoding one path, several known locations are tried in order.
If none resolve to a usable (non-empty) classifier, face detection is
disabled and the app falls back to running on the whole image instead of
crashing.
"""

import glob
import os

import cv2
import numpy as np
from PIL import Image

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _find_cascade_path():
    """Searches several known locations for a usable, non-empty cascade file.
    Returns the first working path, or None if no valid cascade is found
    anywhere on this machine."""
    candidates = []

    # 1. The documented, standard location (works on most OpenCV 3.x/4.x builds)
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, CASCADE_FILENAME))

    # 2. Search the installed cv2 package directory itself, in case the data
    #    folder moved (e.g. under a different subfolder in OpenCV 5 wheels)
    cv2_dir = os.path.dirname(cv2.__file__)
    candidates.extend(glob.glob(os.path.join(cv2_dir, "**", CASCADE_FILENAME), recursive=True))

    # 3. A local copy shipped alongside this project, if you've manually
    #    downloaded one (see README) — checked last as the final fallback
    local_copy = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASCADE_FILENAME)
    candidates.append(local_copy)

    for path in candidates:
        if path and os.path.exists(path):
            test_classifier = cv2.CascadeClassifier(path)
            if not test_classifier.empty():
                return path

    return None


_CASCADE_PATH = _find_cascade_path()
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH) if _CASCADE_PATH else None
FACE_DETECTION_AVAILABLE = _face_cascade is not None


def detect_faces(pil_image: Image.Image):
    """Returns (faces, gray_array). faces is an empty list/array if no
    working cascade classifier was found, so callers can fall back
    gracefully instead of crashing."""
    img_array = np.array(pil_image.convert("L"))
    if not FACE_DETECTION_AVAILABLE:
        return [], img_array
    faces = _face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, img_array


def crop_and_prepare_face(gray_array: np.ndarray, box, target_size=(48, 48)) -> np.ndarray:
    """Crops one face box, resizes to target_size, returns a normalized [0,1] float array."""
    x, y, w, h = box
    face = gray_array[y:y + h, x:x + w]
    face_img = Image.fromarray(face).resize(target_size)
    return np.array(face_img).astype("float32") / 255.0


def prepare_whole_image(pil_image: Image.Image, target_size=(48, 48)) -> np.ndarray:
    """Fallback: no face detected — just resize the whole image to target_size."""
    gray = pil_image.convert("L").resize(target_size)
    return np.array(gray).astype("float32") / 255.0
