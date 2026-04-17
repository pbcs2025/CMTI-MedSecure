"""
Two-stage inference:
1) Medicine detector (medicine vs non_medicine)
2) Authenticity classifier (genuine vs fake)
"""

import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160
MEDICINE_THRESHOLD = 0.70
AUTH_THRESHOLD = 0.70
MAX_CALIBRATION_SAMPLES = 24
STAGE1_OVERRIDE_AUTH_CONF = 0.90


def _load_image(image_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


class MedicineVerifier:
    def __init__(
        self,
        medicine_model_path: str = "model/stage1_medicine_detector.h5",
        authenticity_model_path: str = "model/stage2_authenticity_classifier.h5",
        medicine_threshold: float = MEDICINE_THRESHOLD,
        authenticity_threshold: float = AUTH_THRESHOLD,
        stage1_positive_class: str = "medicine",
    ):
        self.has_stage1 = bool(medicine_model_path) and os.path.exists(medicine_model_path)
        if not os.path.exists(authenticity_model_path):
            raise FileNotFoundError(
                f"Stage 2 model not found: {authenticity_model_path}. Train Stage 2 first."
            )

        self.medicine_model = None
        if self.has_stage1:
            self.medicine_model = tf.keras.models.load_model(medicine_model_path, compile=False)
        self.auth_model = tf.keras.models.load_model(authenticity_model_path, compile=False)
        self.stage1_img_size = (
            int(self.medicine_model.input_shape[1])
            if self.medicine_model is not None and self.medicine_model.input_shape[1]
            else IMG_SIZE
        )
        self.stage2_img_size = int(self.auth_model.input_shape[1]) if self.auth_model.input_shape[1] else IMG_SIZE
        self.medicine_threshold = medicine_threshold
        self.authenticity_threshold = authenticity_threshold
        self.stage1_positive_class = stage1_positive_class
        if self.has_stage1 and self.stage1_positive_class == "auto":
            self.stage1_positive_class = self._infer_stage1_positive_class()
        if self.has_stage1 and self.stage1_positive_class not in {"medicine", "non_medicine"}:
            self.stage1_positive_class = "non_medicine"

    def _sample_images(self, folder: str, max_samples: int = MAX_CALIBRATION_SAMPLES):
        if not os.path.isdir(folder):
            return []
        files = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_exts:
                files.append(path)
        if not files:
            return []
        random.shuffle(files)
        return files[:max_samples]

    def _infer_stage1_positive_class(self) -> str:
        """
        Infer whether stage-1 sigmoid output encodes medicine or non_medicine.
        Uses local dataset samples when available; otherwise falls back safely.
        """
        medicine_paths = self._sample_images("dataset/medicine")
        if not medicine_paths:
            # Fallback: stage-2 classes are still medicine images.
            medicine_paths = self._sample_images("dataset/genuine", max_samples=MAX_CALIBRATION_SAMPLES // 2)
            medicine_paths += self._sample_images("dataset/fake", max_samples=MAX_CALIBRATION_SAMPLES // 2)
        non_medicine_paths = self._sample_images("dataset/non_medicine")
        if not medicine_paths or not non_medicine_paths:
            return "non_medicine"

        medicine_scores = []
        non_medicine_scores = []
        for path in medicine_paths:
            x = _load_image(path, img_size=self.stage1_img_size)
            medicine_scores.append(float(self.medicine_model.predict(x, verbose=0)[0][0]))
        for path in non_medicine_paths:
            x = _load_image(path, img_size=self.stage1_img_size)
            non_medicine_scores.append(float(self.medicine_model.predict(x, verbose=0)[0][0]))

        med_mean = float(np.mean(medicine_scores))
        non_med_mean = float(np.mean(non_medicine_scores))
        return "medicine" if med_mean > non_med_mean else "non_medicine"

    def _stage1_validate(self, x: np.ndarray) -> Tuple[bool, float]:
        raw = float(self.medicine_model.predict(x, verbose=0)[0][0])
        prob_medicine = raw if self.stage1_positive_class == "medicine" else (1.0 - raw)
        is_medicine = prob_medicine >= self.medicine_threshold
        return is_medicine, prob_medicine

    def _stage2_authenticity(self, x: np.ndarray) -> Tuple[str, float, float]:
        raw = float(self.auth_model.predict(x, verbose=0)[0][0])
        if raw >= 0.5:
            label = "fake"
            confidence = raw
        else:
            label = "genuine"
            confidence = 1.0 - raw
        return label, confidence, raw

    def predict(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.has_stage1:
            x_stage1 = _load_image(image_path, img_size=self.stage1_img_size)
            is_medicine, medicine_conf = self._stage1_validate(x_stage1)
            if not is_medicine:
                # Recovery path: if stage-2 is highly confident, do not hard-fail as non-medicine.
                x_stage2 = _load_image(image_path, img_size=self.stage2_img_size)
                label, auth_conf, raw = self._stage2_authenticity(x_stage2)
                if auth_conf >= max(self.authenticity_threshold, STAGE1_OVERRIDE_AUTH_CONF):
                    return {
                        "status": "genuine" if label == "genuine" else "suspicious",
                        "label": "Genuine" if label == "genuine" else "Suspicious",
                        "message": (
                            "Stage 1 had low confidence, but Stage 2 is highly confident. "
                            "Using Stage 2 authenticity verdict."
                        ),
                        "medicine_confidence": round(medicine_conf, 4),
                        "confidence": round(auth_conf, 4),
                        "raw_score": round(raw, 4),
                        "risk_level": "low" if label == "genuine" and auth_conf >= 0.85 else ("moderate" if label == "genuine" else "high"),
                    }
                return {
                    "status": "not_medicine",
                    "label": "Not a medicine image",
                    "message": "Stage 1 rejected the image as non-medicine.",
                    "medicine_confidence": round(medicine_conf, 4),
                    "confidence": round(medicine_conf, 4),
                    "risk_level": "unknown",
                    "raw_score": 0.0,
                }
        else:
            medicine_conf = 1.0

        x_stage2 = _load_image(image_path, img_size=self.stage2_img_size)
        label, auth_conf, raw = self._stage2_authenticity(x_stage2)
        if auth_conf < self.authenticity_threshold:
            return {
                "status": "unverified",
                "label": "Unverified result",
                "message": (
                    "Stage 2 confidence is below threshold. "
                    "Please upload a clearer medicine image."
                ),
                "medicine_confidence": round(medicine_conf, 4),
                "confidence": round(auth_conf, 4),
                "raw_score": round(raw, 4),
                "risk_level": "moderate",
            }

        return {
            "status": "genuine" if label == "genuine" else "suspicious",
            "label": "Genuine" if label == "genuine" else "Suspicious",
            "message": "Prediction completed." if not self.has_stage1 else "Two-stage prediction completed.",
            "medicine_confidence": round(medicine_conf, 4),
            "confidence": round(auth_conf, 4),
            "raw_score": round(raw, 4),
            "risk_level": "low" if label == "genuine" and auth_conf >= 0.85 else ("moderate" if label == "genuine" else "high"),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage medicine inference.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--medicine-model", default="model/stage1_medicine_detector.h5")
    parser.add_argument("--auth-model", default="model/stage2_authenticity_classifier.h5")
    parser.add_argument("--medicine-threshold", type=float, default=MEDICINE_THRESHOLD)
    parser.add_argument("--auth-threshold", type=float, default=AUTH_THRESHOLD)
    parser.add_argument(
        "--stage1-positive-class",
        choices=["auto", "medicine", "non_medicine"],
        default="medicine",
        help=(
            "Interpretation of stage1 sigmoid output. "
            "'auto' infers from dataset/medicine and dataset/non_medicine."
        ),
    )
    args = parser.parse_args()

    verifier = MedicineVerifier(
        medicine_model_path=args.medicine_model,
        authenticity_model_path=args.auth_model,
        medicine_threshold=args.medicine_threshold,
        authenticity_threshold=args.auth_threshold,
        stage1_positive_class=args.stage1_positive_class,
    )
    result = verifier.predict(args.image)
    print(result)
