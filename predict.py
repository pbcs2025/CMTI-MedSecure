"""
Two-stage inference:
1) Medicine detector (medicine vs non_medicine)
2) Authenticity classifier (genuine vs fake)
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160
MEDICINE_THRESHOLD = 0.70
AUTH_THRESHOLD = 0.70


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
        self.img_size = int(self.auth_model.input_shape[1]) if self.auth_model.input_shape[1] else IMG_SIZE
        self.medicine_threshold = medicine_threshold
        self.authenticity_threshold = authenticity_threshold

    def _stage1_validate(self, x: np.ndarray) -> Tuple[bool, float]:
        raw = float(self.medicine_model.predict(x, verbose=0)[0][0])
        prob_medicine = 1.0 - raw
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

        x = _load_image(image_path, img_size=self.img_size)

        if self.has_stage1:
            is_medicine, medicine_conf = self._stage1_validate(x)
            if not is_medicine:
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

        label, auth_conf, raw = self._stage2_authenticity(x)
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
    args = parser.parse_args()

    verifier = MedicineVerifier(
        medicine_model_path=args.medicine_model,
        authenticity_model_path=args.auth_model,
        medicine_threshold=args.medicine_threshold,
        authenticity_threshold=args.auth_threshold,
    )
    result = verifier.predict(args.image)
    print(result)
