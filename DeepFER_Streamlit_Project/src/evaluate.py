"""
Evaluates the trained model on the held-out test set and saves:
  - a JSON classification report (per-class precision/recall/F1)
  - a confusion matrix (as a .npy array)

These are what the Streamlit dashboard's "Model Insights" tab reads —
evaluation is done once here, not inside the app, so the app stays fast.

    python src/evaluate.py
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_pipeline import build_datasets, prepare_dataset

import tensorflow as tf


def main():
    print("Loading model and test set...")
    model = tf.keras.models.load_model(config.MODEL_PATH)
    _, _, test_ds, class_names = build_datasets()
    test_ds_prepared = prepare_dataset(test_ds)

    y_true, y_pred = [], []
    for images, labels in test_ds_prepared:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    np.save(config.CONFUSION_MATRIX_PATH, cm)

    print(f"Overall accuracy: {report['accuracy']:.4f}")
    print("\nPer-class F1-score:")
    for name in class_names:
        print(f"  {name:10s}: {report[name]['f1-score']:.3f}")
    print(f"\nSaved evaluation report to {config.EVAL_REPORT_PATH}")
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
