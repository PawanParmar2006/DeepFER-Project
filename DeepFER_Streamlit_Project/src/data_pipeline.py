"""
Data pipeline for DeepFER: builds train/val/test tf.data datasets from the
folder structure, computes class weights, and saves the label map used by
both training and the Streamlit app.
"""

import json
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def count_images_per_class(directory: str) -> dict:
    counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            counts[class_name] = len([f for f in os.listdir(class_dir) if not f.startswith(".")])
    return counts


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, validation_split=config.VALIDATION_SPLIT, subset="training",
        seed=config.SEED, image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        color_mode=config.COLOR_MODE, batch_size=config.BATCH_SIZE, label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, validation_split=config.VALIDATION_SPLIT, subset="validation",
        seed=config.SEED, image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        color_mode=config.COLOR_MODE, batch_size=config.BATCH_SIZE, label_mode="categorical",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR, shuffle=False, image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        color_mode=config.COLOR_MODE, batch_size=config.BATCH_SIZE, label_mode="categorical",
    )

    class_names = train_ds.class_names
    if class_names != config.CLASS_NAMES:
        raise ValueError(
            f"Folder-derived class order {class_names} != config.CLASS_NAMES "
            f"{config.CLASS_NAMES}. Fix your folder names or update config.py."
        )
    return train_ds, val_ds, test_ds, class_names


def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def prepare_dataset(ds: tf.data.Dataset, augment_layer=None, training: bool = False):
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment_layer is not None:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def compute_class_weights(directory: str, class_names: list) -> dict:
    counts = count_images_per_class(directory)
    total = sum(counts.values())
    n_classes = len(class_names)
    return {idx: total / (n_classes * counts.get(name, 1)) for idx, name in enumerate(class_names)}


def save_label_map(class_names: list):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    label_map = {str(idx): name for idx, name in enumerate(class_names)}
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map to {config.LABEL_MAP_PATH}")


if __name__ == "__main__":
    print("Train class counts:")
    for name, count in count_images_per_class(config.TRAIN_DIR).items():
        print(f"  {name:10s}: {count}")
    print("\nTest class counts:")
    for name, count in count_images_per_class(config.TEST_DIR).items():
        print(f"  {name:10s}: {count}")
