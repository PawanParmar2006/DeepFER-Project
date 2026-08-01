"""
Trains the DeepFER CNN.

    python src/train.py

Uses class weights (imbalance correction), data augmentation, and
EarlyStopping/ModelCheckpoint/ReduceLROnPlateau callbacks.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_pipeline import (
    build_datasets, build_augmentation_layer, prepare_dataset,
    compute_class_weights, save_label_map,
)
from model import build_model

from tensorflow.keras import optimizers, callbacks


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("Building datasets...")
    train_ds, val_ds, test_ds, class_names = build_datasets()
    save_label_map(class_names)

    augment_layer = build_augmentation_layer()
    train_ds = prepare_dataset(train_ds, augment_layer=augment_layer, training=True)
    val_ds = prepare_dataset(val_ds)

    print("Computing class weights...")
    class_weights = compute_class_weights(config.TRAIN_DIR, class_names)
    for idx, name in enumerate(class_names):
        print(f"  {name:10s}: weight = {class_weights[idx]:.3f}")

    print("Building model...")
    model = build_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.summary()

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                                 restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=config.MODEL_PATH, monitor="val_loss",
                                   save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
    ]

    print("Starting training...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=config.EPOCHS,
        class_weight=class_weights, callbacks=callback_list,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"Saved training history to {config.HISTORY_PATH}")
    print(f"Best model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
