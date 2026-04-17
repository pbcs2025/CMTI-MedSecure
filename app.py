"""
app.py  (v2)
------------
Flask backend for the Counterfeit Medicine Detection web app.
Uses predict_v2.py which handles non-medicine images and low-confidence cases.

Usage:
    pip install flask
    python app.py
    Open: http://localhost:5000
"""

import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import MedicineVerifier
from blockchain_service import (
    create_result_object,
    generate_verification_hash,
    store_on_chain,
)

UPLOAD_FOLDER  = "uploads"
ALLOWED_EXTS   = {"jpg", "jpeg", "png", "webp", "avif"}
MAX_CONTENT_MB = 16

def find_model():
    model_dir = "model"
    if not os.path.isdir(model_dir):
        return None

    # 1) Prefer native Keras format
    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    if keras_files:
        keras_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        return os.path.join(model_dir, keras_files[0])

    # 2) Fallback to SavedModel export directories
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

    # 3) Last resort: legacy full-model .h5 (ignore weights-only files)
    h5_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".h5") and not f.endswith(".weights.h5")
    ]
    if h5_files:
        h5_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        return os.path.join(model_dir, h5_files[0])

    return None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = find_model()
if not model_path:
    print("[ERROR] No trained model found. Run train.py first.")
    sys.exit(1)

print(f"[INFO] Loading model: {model_path}")
verifier = MedicineVerifier(model_path)
print("[INFO] Server ready.\n")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use JPG, PNG or WEBP"}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = verifier.predict(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": str(e)}), 500

    os.remove(filepath)

    return jsonify({
        "status":     result["status"],
        "label":      result["label"],
        "confidence": round(result["confidence"] * 100, 1),
        "risk_level": result["risk_level"],
        "raw_score":  result["raw_score"],
        "message":    result["message"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model_path})


@app.route("/store-result", methods=["POST"])
def store_result():
    """
    Store an already computed verification result on blockchain.
    Expected JSON input:
    {
      "prediction": "Genuine" | "Suspicious",
      "confidence": 0.93,
      "risk_level": "LOW" | "HIGH"   # optional; auto-derived if absent
    }
    """
    data = request.get_json(silent=True) or {}
    prediction = data.get("prediction")
    confidence = data.get("confidence")
    risk_level = data.get("risk_level")

    if prediction is None or confidence is None:
        return jsonify({"error": "prediction and confidence are required"}), 400

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return jsonify({"error": "confidence must be numeric"}), 400

    result_obj = create_result_object(
        prediction=prediction,
        confidence=confidence,
        risk_level=risk_level,
    )
    verification_hash = generate_verification_hash(result_obj)

    try:
        tx_hash = store_on_chain(
            image_id=result_obj["image_id"],
            hash_val=verification_hash,
            risk=result_obj["risk_level"],
            timestamp=result_obj["timestamp"],
        )
    except Exception as e:
        return jsonify({"error": f"blockchain store failed: {e}"}), 500

    return jsonify(
        {
            "blockchain_tx_hash": tx_hash,
            "verification_hash": verification_hash,
            "image_id": result_obj["image_id"],
            "timestamp": result_obj["timestamp"],
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  MediScan — Counterfeit Medicine Detector")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)