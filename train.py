"""
train.py
--------
Two-stage training pipeline:
  Stage 1: medicine vs non_medicine
  Stage 2: genuine vs fake
"""

import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications import MobileNetV2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VALID_EXTS)


def bootstrap_medicine_folder():
    """
    If dataset/medicine is missing, populate it from existing genuine+fake images.
    This avoids hard crash for users migrating from old project structure.
    """
    medicine_dir = "dataset/medicine"
    if os.path.isdir(medicine_dir) and _count_images(medicine_dir) > 0:
        return

    os.makedirs(medicine_dir, exist_ok=True)
    copied = 0
    for src_dir in ["dataset/genuine", "dataset/fake"]:
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            ext = os.path.splitext(name)[1].lower()
            if ext not in VALID_EXTS:
                continue
            src = os.path.join(src_dir, name)
            dst = os.path.join(medicine_dir, f"{os.path.basename(src_dir)}_{name}")
            if os.path.exists(dst):
                continue
            with open(src, "rb") as rf, open(dst, "wb") as wf:
                wf.write(rf.read())
            copied += 1
    if copied > 0:
        print(f"Bootstrap: copied {copied} images into dataset/medicine")


def collect_paths(class_a_dir: str, class_b_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    for idx, folder in enumerate([class_a_dir, class_b_dir]):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        paths.extend(files)
        labels.extend([idx] * len(files))
        print(f"{folder}: {len(files)} images")

    return np.array(paths), np.array(labels, dtype=np.int32)


def stratified_cap(paths: np.ndarray, labels: np.ndarray, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce dataset size with class balance preserved."""
    if max_samples <= 0 or len(paths) <= max_samples:
        return paths, labels
    x_keep, _, y_keep, _ = train_test_split(
        paths, labels, train_size=max_samples, stratify=labels, random_state=SEED
    )
    return x_keep, y_keep


def decode_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def make_dataset(paths, labels, img_size, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(
        lambda p, y: (decode_image(p, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_binary_model(img_size: int, backbone: str = "mobilenet") -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, 3))
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )(inputs)
    x = layers.Lambda(lambda t: t * 2.0 - 1.0)(aug)

    if backbone == "resnet":
        base = keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
    else:
        base = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )

    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


def fit_one_model(model, train_ds, val_ds, epochs: int, early_stop_patience: int = 5):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=early_stop_patience,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-7),
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=1)


def train_stage(
    stage_name: str,
    dir_a: str,
    dir_b: str,
    output_path: str,
    img_size: int,
    batch: int,
    epochs: int,
    backbone: str,
    early_stop_patience: int = 5,
    max_samples: int = 0,
):
    print(f"\n===== {stage_name} =====")
    paths, labels = collect_paths(dir_a, dir_b)
    paths, labels = stratified_cap(paths, labels, max_samples=max_samples)
    if max_samples > 0:
        print(f"Capped dataset to {len(paths)} samples for faster training.")
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    train_ds = make_dataset(x_train, y_train, img_size, batch, training=True)
    val_ds = make_dataset(x_val, y_val, img_size, batch)
    test_ds = make_dataset(x_test, y_test, img_size, batch)

    model = build_binary_model(img_size, backbone=backbone)
    fit_one_model(model, train_ds, val_ds, epochs, early_stop_patience=early_stop_patience)

    loss, acc, auc = model.evaluate(test_ds, verbose=0)
    print(f"{stage_name} test -> loss={loss:.4f}, acc={acc:.4f}, auc={auc:.4f}")
    model.save(output_path)
    print(f"Saved model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train two-stage medicine pipeline.")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=4)
    parser.add_argument("--epochs-stage2", type=int, default=25)
    parser.add_argument("--patience-stage1", type=int, default=2)
    parser.add_argument("--patience-stage2", type=int, default=5)
    parser.add_argument(
        "--max-samples-stage1",
        type=int,
        default=6000,
        help="Cap Stage 1 images for faster runs. Set 0 to use all data.",
    )
    parser.add_argument("--backbone", choices=["mobilenet", "resnet"], default="mobilenet")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 if non_medicine data is not ready.")
    parser.add_argument("--skip-stage2", action="store_true", help="Train only Stage 1 and skip Stage 2.")
    parser.add_argument(
        "--force-stage2",
        action="store_true",
        help="Retrain Stage 2 even if model/stage2_authenticity_classifier.h5 already exists.",
    )
    args = parser.parse_args()

    bootstrap_medicine_folder()

    can_train_stage1 = (
        os.path.isdir("dataset/medicine")
        and os.path.isdir("dataset/non_medicine")
        and _count_images("dataset/medicine") > 0
        and _count_images("dataset/non_medicine") > 0
        and not args.skip_stage1
    )
    if can_train_stage1:
        train_stage(
            stage_name="Stage 1 (medicine vs non_medicine)",
            dir_a="dataset/medicine",
            dir_b="dataset/non_medicine",
            output_path="model/stage1_medicine_detector.h5",
            img_size=args.img_size,
            batch=args.batch,
            epochs=args.epochs_stage1,
            backbone=args.backbone,
            early_stop_patience=args.patience_stage1,
            max_samples=args.max_samples_stage1,
        )
    else:
        print("\nSkipping Stage 1 training.")
        print("Reason: dataset/non_medicine is missing/empty or --skip-stage1 used.")
        print("Add non-medicine images to dataset/non_medicine, then re-run train.py for full two-stage support.")

    stage2_model_path = "model/stage2_authenticity_classifier.h5"
    if args.skip_stage2:
        print("\nSkipping Stage 2 training.")
        print("Reason: --skip-stage2 used.")
    elif os.path.exists(stage2_model_path) and not args.force_stage2:
        print(f"\nSkipping Stage 2 training.")
        print(f"Reason: existing model found at {stage2_model_path}.")
        print("Use --force-stage2 to retrain this stage.")
    else:
        train_stage(
            stage_name="Stage 2 (genuine vs fake)",
            dir_a="dataset/genuine",
            dir_b="dataset/fake",
            output_path=stage2_model_path,
            img_size=args.img_size,
            batch=args.batch,
            epochs=args.epochs_stage2,
            backbone=args.backbone,
            early_stop_patience=args.patience_stage2,
            max_samples=0,
        )

    print("\nTraining complete.")
    print("Generated models:")
    print("- model/stage1_medicine_detector.h5")
    print("- model/stage2_authenticity_classifier.h5")


if __name__ == "__main__":
    main()
