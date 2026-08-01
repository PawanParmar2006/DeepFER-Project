"""
CNN architecture for DeepFER — a 3-block VGG-style CNN tuned for small
(48x48) grayscale facial images. See project README for design rationale.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from tensorflow.keras import layers, models, regularizers


def build_model() -> models.Model:
    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    x = inputs

    # Block 1: 48x48 -> 24x24
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

    # Block 2: 24x24 -> 12x12
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.3)(x)

    # Block 3: 12x12 -> 6x6
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                       name=config.LAST_CONV_LAYER)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.35)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="DeepFER_CNN")


if __name__ == "__main__":
    build_model().summary()
