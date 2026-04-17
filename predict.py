"""
predict.py
----------
Loads the trained model and runs inference on a single image.
This is the function your Flask / FastAPI backend will import.

v2 Changes:
  - Confidence threshold gate: below 65% returns "unverified" instead of
    a misleading genuine/fake label
  - Image relevance check: detects non-packaging images using pixel
    variance and color distribution heuristics
  - Four result states: genuine / suspicious / unverified / not_medicine

Usage (standalone test):
    python predict.py --image path/to/medicine.jpg --model model/yourmodel.h5

Usage (import in backend):
    from predict import MedicineVerifier
    verifier = MedicineVerifier("model/counterfeit_mobilenetv2_e30_b32.h5")
    result   = verifier.predict("uploads/scan.jpg")
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageStat

IMG_SIZE = 224

# ── Confidence thresholds ─────────────────────────────────────────────────────
GENUINE_LOW_THRESHOLD      = 0.90   # ≥ 90% genuine  → LOW risk
GENUINE_MODERATE_THRESHOLD = 0.70   # ≥ 70% genuine  → MODERATE risk
CONFIDENCE_MIN_THRESHOLD   = 0.65   # < 65% either   → UNVERIFIED
FAKE_HIGH_THRESHOLD        = 0.75   # ≥ 75% fake     → HIGH risk

# ── Image relevance thresholds ────────────────────────────────────────────────
# Medicine packaging tends to have:
#   - Moderate variance (not too uniform, not too chaotic)
#   - Visible text/edges → decent edge energy
MIN_VARIANCE   = 200    # Too uniform = blank wall, plain surface, etc.
MAX_VARIANCE   = 6000   # Too chaotic = random scene, no clear subject
MIN_BRIGHTNESS = 30     # Too dark to see anything
MAX_BRIGHTNESS = 245    # Completely washed out


class MedicineVerifier:

    def __init__(self, model_path: str,
                 class_index_path: str = "model/class_indices.json"):
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)

        if os.path.exists(class_index_path):
            with open(class_index_path) as f:
                self.class_index = json.load(f)
        else:
            self.class_index = {"0": "genuine", "1": "fake"}

        print("Model ready.")

    def _load_model(self, model_path: str):
        """
        Load either:
          - native Keras model file (.keras/.h5), or
          - TF SavedModel directory (via TFSMLayer wrapper).
        """
        if os.path.isdir(model_path):
            saved_pb = os.path.join(model_path, "saved_model.pb")
            if not os.path.exists(saved_pb):
                raise ValueError(f"Invalid SavedModel directory: {model_path}")

            tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serve")
            inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            outputs = tfsm_layer(inputs)
            return tf.keras.Model(inputs=inputs, outputs=outputs, name="saved_model_wrapper")

        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except ValueError as e:
            if model_path.endswith(".h5") and "No model config found" in str(e):
                raise ValueError(
                    f"{model_path} is not a full Keras model file. "
                    "It is likely a weights-only or incomplete H5 file. "
                    "Use a .keras model or a SavedModel directory."
                ) from e
            raise

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, image_path: str) -> tuple:
        """
        Load and resize image. Returns (tensor, pil_image).
        PIL image is used for the relevance check.
        """
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0), img

    # ── Image relevance check ──────────────────────────────────────────────────

    def _is_likely_medicine_image(self, img: Image.Image) -> tuple:
        """
        Quick sanity check — does this image look like it could be
        medicine packaging at all?

        Returns (is_valid: bool, reason: str)

        Checks:
          1. Brightness — too dark or too bright → reject
          2. Variance   — too uniform (blank wall) or too chaotic
                          (complex natural scene) → reject
          3. Aspect ratio — extremely tall/wide images are unlikely
                            to be medicine packaging photos
        """
        # Resize to small for fast analysis
        small = img.resize((64, 64))
        arr   = np.array(small, dtype=np.float32)

        # 1. Brightness check
        brightness = arr.mean()
        if brightness < MIN_BRIGHTNESS:
            return False, "Image is too dark to analyse"
        if brightness > MAX_BRIGHTNESS:
            return False, "Image is too bright or washed out"

        # 2. Variance check
        variance = arr.var()
        if variance < MIN_VARIANCE:
            return False, "Image appears to be a blank or uniform surface — not medicine packaging"
        if variance > MAX_VARIANCE:
            return False, "Image appears to be a complex scene — please photograph only the medicine packaging"

        # 3. Aspect ratio check
        w, h = img.size
        ratio = max(w, h) / min(w, h)
        if ratio > 5.0:
            return False, "Unusual image dimensions — please take a standard photo of the packaging"

        return True, "ok"

    # ── Risk level ─────────────────────────────────────────────────────────────

    def _risk_level(self, confidence: float, label: str) -> str:
        if label == "genuine":
            if confidence >= GENUINE_LOW_THRESHOLD:
                return "low"
            elif confidence >= GENUINE_MODERATE_THRESHOLD:
                return "moderate"
            else:
                return "high"
        else:
            if confidence >= FAKE_HIGH_THRESHOLD:
                return "high"
            else:
                return "moderate"

    # ── Main predict ───────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image.

        Returns dict with keys:
            status      : "genuine" | "suspicious" | "unverified" | "not_medicine"
            label       : human readable label
            confidence  : float 0-1
            risk_level  : "low" | "moderate" | "high" | "unknown"
            raw_score   : raw sigmoid output
            message     : plain English explanation for the user
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ── Step 1: Load image ─────────────────────────────────────────────────
        x, pil_img = self._preprocess(image_path)

        # ── Step 2: Relevance check ────────────────────────────────────────────
        is_valid, reason = self._is_likely_medicine_image(pil_img)
        if not is_valid:
            return {
                "status":     "not_medicine",
                "label":      "Not Medicine Packaging",
                "confidence": 0.0,
                "risk_level": "unknown",
                "raw_score":  0.0,
                "message":    f"{reason}. Please upload a clear photo of the medicine box, strip, or bottle label."
            }

        # ── Step 3: Model inference ────────────────────────────────────────────
        raw_score = float(self.model.predict(x, verbose=0)[0][0])

        if raw_score >= 0.5:
            label      = "fake"
            confidence = raw_score
        else:
            label      = "genuine"
            confidence = 1.0 - raw_score

        # ── Step 4: Confidence gate ────────────────────────────────────────────
        # If model is not confident enough either way → unverified
        if confidence < CONFIDENCE_MIN_THRESHOLD:
            return {
                "status":     "unverified",
                "label":      "Unable to Verify",
                "confidence": round(confidence, 4),
                "risk_level": "moderate",
                "raw_score":  round(raw_score, 4),
                "message":    (
                    "The AI could not confidently verify this medicine. "
                    "This may be because the medicine is not in our training database yet, "
                    "the photo quality is low, or the packaging is partially obscured. "
                    "Please verify with your pharmacist."
                )
            }

        # ── Step 5: Normal result ──────────────────────────────────────────────
        risk   = self._risk_level(confidence, label)
        status = "genuine" if label == "genuine" else "suspicious"

        messages = {
            ("genuine", "low"): (
                "Packaging visual patterns match genuine medicine characteristics. "
                "Appears authentic. Always check the expiry date and tamper-evident seal before use."
            ),
            ("genuine", "moderate"): (
                "Packaging appears genuine but confidence is moderate. "
                "This medicine may not be in our full training database yet. "
                "Verify the batch number, expiry date, and seal with your pharmacist."
            ),
            ("genuine", "high"): (
                "The model leaned towards genuine but with low confidence. "
                "Exercise caution. Have this medicine verified by a pharmacist before use."
            ),
            ("suspicious", "moderate"): (
                "Some packaging inconsistencies were detected. "
                "This could be a counterfeit product or a medicine variant not yet in our database. "
                "Do not consume without pharmacist verification."
            ),
            ("suspicious", "high"): (
                "High likelihood of counterfeit packaging detected. "
                "Do NOT consume this medicine. Return it to the pharmacy "
                "and report to your nearest drug regulatory authority immediately."
            ),
        }

        message = messages.get(
            (status, risk),
            "Please consult your pharmacist for verification."
        )

        return {
            "status":     status,
            "label":      "Genuine Medicine" if status == "genuine" else "Suspicious Package",
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "raw_score":  round(raw_score, 4),
            "message":    message
        }


# ── Standalone test ───────────────────────────────────────────────────────────

def _find_latest_model():
    model_dir = "model"
    if not os.path.isdir(model_dir):
        return None

    # Prefer native Keras format
    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    if keras_files:
        keras_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        return os.path.join(model_dir, keras_files[0])

    # Fallback to SavedModel directories
    saved_root = os.path.join(model_dir, "saved_model")
    if os.path.isdir(saved_root):
        saved_dirs = []
        for name in os.listdir(saved_root):
            path = os.path.join(saved_root, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
                saved_dirs.append(path)
        if saved_dirs:
            saved_dirs.sort(key=os.path.getmtime, reverse=True)
            return saved_dirs[0]

    # Final fallback: legacy full-model .h5
    h5_files = [f for f in os.listdir(model_dir) if f.endswith(".h5") and not f.endswith(".weights.h5")]
    if h5_files:
        h5_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        return os.path.join(model_dir, h5_files[0])

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--model",  default=None)
    args = parser.parse_args()

    
    model_path = args.model or _find_latest_model()
    if not model_path:
        print("❌  No model found. Run train.py first.")
        exit(1)

    verifier = MedicineVerifier(model_path)
    result   = verifier.predict(args.image)

    print("\n── Prediction Result ─────────────────────────────────")
    print(f"  Image      : {args.image}")
    print(f"  Status     : {result['status'].upper()}")
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"  Risk Level : {result['risk_level'].upper()}")
    print(f"  Message    : {result['message']}")
    print("──────────────────────────────────────────────────────\n")