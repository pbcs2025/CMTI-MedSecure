"""
train.py
--------
Full training pipeline for the Counterfeit Medicine Detection model.

Architecture : MobileNetV2 (ImageNet pretrained) + custom classification head
Framework    : TensorFlow / Keras
Input        : dataset/genuine/  and  dataset/fake/
Output       : model/saved_model/       ← full SavedModel for backend serving
               model/saved_model.h5     ← single-file Keras backup
               model/checkpoints/       ← best weights during training
               logs/training/           ← TensorBoard logs

Usage:
    pip install tensorflow scikit-learn matplotlib seaborn
    python train.py

Optional flags:
    --epochs     50          (default: 30)
    --batch      32          (default: 32)
    --img-size   224         (default: 224)
    --fine-tune               Enable second-pass fine-tuning of top conv layers
    --no-plot                 Skip saving evaluation plots
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
GENUINE_DIR     = "dataset/genuine"
FAKE_DIR        = "dataset/fake"
CHECKPOINT_DIR  = "model/checkpoints"
SAVEDMODEL_DIR  = "model/saved_model"
LOG_DIR         = "logs/training"
PLOTS_DIR       = "model/plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVEDMODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(PLOTS_DIR,      exist_ok=True)

CLASS_NAMES   = ["genuine", "fake"]   # 0 = genuine, 1 = fake
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_paths_and_labels(genuine_dir: str, fake_dir: str):
    """Return parallel lists of file paths and integer labels."""
    paths, labels = [], []

    for label_idx, folder in enumerate([genuine_dir, fake_dir]):
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Folder not found: '{folder}'\n"
                "Run setup_dataset.py first, then populate dataset/genuine/ "
                "with images, then run generate_fakes.py."
            )
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        paths.extend(files)
        labels.extend([label_idx] * len(files))
        print(f"   {CLASS_NAMES[label_idx]:>10} : {len(files)} images")

    return np.array(paths), np.array(labels)


def decode_image(path: str, img_size: int) -> tf.Tensor:
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img  = tf.image.resize(img, [img_size, img_size])
    img  = tf.cast(img, tf.float32) / 255.0   # normalise to [0, 1]
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AUGMENTATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# Applied only during training to artificially expand diversity.

def get_augmentation_layer():
    """
    Returns a Sequential augmentation sub-model.
    These layers are active only during model.fit (training=True).
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),            # ±12% rotation
        layers.RandomZoom(0.15),                # ±15% zoom
        layers.RandomTranslation(0.1, 0.1),     # shift up to 10%
        layers.RandomBrightness(0.2),           # ±20% brightness
        layers.RandomContrast(0.2),             # ±20% contrast
    ], name="augmentation")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(img_size: int, dropout_rate: float = 0.4):
    """
    Transfer learning model:
        Input → Augmentation → MobileNetV2 (frozen) → GlobalAvgPool
              → Dense(256, ReLU) → Dropout → Dense(1, Sigmoid)

    Output: single sigmoid neuron
        < 0.5  →  Genuine
        ≥ 0.5  →  Fake / Suspicious
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_input")

    # Augmentation (only active during training)
    x = get_augmentation_layer()(inputs)

    # MobileNetV2 preprocessing (scales [0,1] → [-1, 1])
    x = layers.Lambda(
        lambda t: t * 2.0 - 1.0,
        name="mobilenet_preprocess"
    )(x)

    # Pretrained backbone – all layers frozen initially
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="CounterfeitDetector")
    return model, base_model


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def make_tf_dataset(paths, labels, img_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(
        lambda p, l: (decode_image(p, img_size), l),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_callbacks(run_id: str):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_best.weights.h5")
    return [
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, run_id),
            histogram_freq=1
        ),
    ]


def phase1_train(model, train_ds, val_ds, epochs, run_id):
    """Phase 1: train only the classification head (backbone frozen)."""
    print("\n── Phase 1: Training classification head (backbone frozen) ──\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(f"{run_id}_phase1"),
        verbose=1
    )
    return history


def phase2_finetune(model, base_model, train_ds, val_ds, epochs, run_id):
    """
    Phase 2 (optional): unfreeze top N layers of MobileNetV2 and fine-tune
    with a very low learning rate so pretrained weights are not destroyed.
    """
    print("\n── Phase 2: Fine-tuning top layers of MobileNetV2 ──\n")

    # Unfreeze the last 30 layers of the backbone
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 100× lower LR
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(f"{run_id}_phase2"),
        verbose=1
    )
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_ds, test_labels, save_plots: bool, run_id: str):
    print("\n── Evaluation on held-out test set ──\n")

    # Raw predictions
    y_prob = model.predict(test_ds, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Classification report
    report = classification_report(
        test_labels, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    auc = roc_auc_score(test_labels, y_prob)
    print(f"   ROC-AUC : {auc:.4f}")

    # Save report to file
    report_path = os.path.join(PLOTS_DIR, f"{run_id}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")
    print(f"\n   Report saved → {report_path}")

    if save_plots:
        _plot_confusion_matrix(test_labels, y_pred, run_id)
        _plot_roc_curve(test_labels, y_prob, auc, run_id)

    return auc


def _plot_confusion_matrix(y_true, y_pred, run_id):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    path = os.path.join(PLOTS_DIR, f"{run_id}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {path}")


def _plot_roc_curve(y_true, y_prob, auc, run_id):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#E63946", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    path = os.path.join(PLOTS_DIR, f"{run_id}_roc_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ROC curve saved → {path}")


def _plot_training_history(history_list, run_id):
    """Combine Phase 1 (+ optional Phase 2) training curves."""
    merged = {"accuracy": [], "val_accuracy": [],
              "loss":     [], "val_loss":     [],
              "auc":      [], "val_auc":      []}
    for h in history_list:
        for key in merged:
            if key in h.history:
                merged[key].extend(h.history[key])

    epochs = range(1, len(merged["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (train_key, val_key, title) in zip(axes, [
        ("loss",     "val_loss",     "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("auc",      "val_auc",      "AUC"),
    ]):
        ax.plot(epochs, merged[train_key], label="Train",  color="#457B9D")
        ax.plot(epochs, merged[val_key],   label="Val",    color="#E63946")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold", y=1.02)
    path = os.path.join(PLOTS_DIR, f"{run_id}_training_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Training history saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_model(model, run_id: str):
    """
    Export in two formats:
      1. TF SavedModel  – recommended for backend (TF Serving / tf.saved_model.load)
      2. .h5 file       – single-file Keras backup
    """
    # TF SavedModel
    saved_path = os.path.join(SAVEDMODEL_DIR, run_id)
    model.export(saved_path)
    print(f"\n✅  SavedModel exported → {saved_path}/")

    # .h5 backup
    h5_path = os.path.join("model", f"{run_id}.h5")
    model.save(h5_path)
    print(f"✅  Keras .h5 backup   → {h5_path}")

    # Save class index mapping for backend use
    import json
    class_index = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    idx_path = os.path.join("model", "class_indices.json")
    with open(idx_path, "w") as f:
        json.dump(class_index, f, indent=2)
    print(f"✅  Class index map    → {idx_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    run_id = f"counterfeit_mobilenetv2_e{args.epochs}_b{args.batch}"
    img_size = args.img_size

    print("\n" + "=" * 60)
    print("  COUNTERFEIT MEDICINE DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Image size : {img_size}×{img_size}")
    print(f"  Batch size : {args.batch}")
    print(f"  Epochs     : {args.epochs}  (Phase 1)")
    print(f"  Fine-tune  : {'Yes — Phase 2 enabled' if args.fine_tune else 'No'}")
    print(f"  Run ID     : {run_id}")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("\n[1/6] Loading image paths ...")
    paths, labels = load_paths_and_labels(GENUINE_DIR, FAKE_DIR)
    print(f"      Total: {len(paths)} images across {len(CLASS_NAMES)} classes")

    # ── 2. Split: 70% train / 15% val / 15% test ──────────────────────────
    print("\n[2/6] Splitting dataset (70/15/15) ...")
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"      Train : {len(X_train)}   Val : {len(X_val)}   Test : {len(X_test)}")

    # ── 3. Build tf.data pipelines ─────────────────────────────────────────
    print("\n[3/6] Building tf.data pipelines ...")
    train_ds = make_tf_dataset(X_train, y_train, img_size, args.batch, shuffle=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   img_size, args.batch)
    test_ds  = make_tf_dataset(X_test,  y_test,  img_size, args.batch)

    # ── 4. Build model ─────────────────────────────────────────────────────
    print("\n[4/6] Building MobileNetV2 model ...")
    model, base_model = build_model(img_size)
    model.summary(line_length=80)

    # ── 5. Train ───────────────────────────────────────────────────────────
    print("\n[5/6] Training ...")
    histories = []

    h1 = phase1_train(model, train_ds, val_ds, args.epochs, run_id)
    histories.append(h1)

    if args.fine_tune:
        ft_epochs = max(10, args.epochs // 3)
        h2 = phase2_finetune(model, base_model, train_ds, val_ds, ft_epochs, run_id)
        histories.append(h2)

    if not args.no_plot:
        _plot_training_history(histories, run_id)

    # ── 6. Evaluate & export ───────────────────────────────────────────────
    print("\n[6/6] Evaluating and exporting ...")
    evaluate_model(model, test_ds, y_test, not args.no_plot, run_id)
    export_model(model, run_id)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
  Model files:
    SavedModel  →  model/saved_model/{run_id}/
    Keras .h5   →  model/{run_id}.h5
    Class map   →  model/class_indices.json

  To use in your Flask/FastAPI backend:
    import tensorflow as tf
    model = tf.saved_model.load("model/saved_model/{run_id}")
    # OR
    model = tf.keras.models.load_model("model/{run_id}.h5")

  To view training curves:
    tensorboard --logdir logs/training
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the counterfeit medicine detection model."
    )
    parser.add_argument("--epochs",    type=int,  default=30,
                        help="Number of training epochs (Phase 1)")
    parser.add_argument("--batch",     type=int,  default=32,
                        help="Batch size")
    parser.add_argument("--img-size",  type=int,  default=224,
                        help="Image input size (default: 224)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Enable Phase 2 fine-tuning of top MobileNetV2 layers")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving evaluation plots")
    args = parser.parse_args()
    main(args)
