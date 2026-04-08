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

UPLOAD_FOLDER  = "uploads"
ALLOWED_EXTS   = {"jpg", "jpeg", "png", "webp", "avif"}
MAX_CONTENT_MB = 16

def find_model():
    h5_dir = "model"
    if os.path.isdir(h5_dir):
        files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
        if files:
            files.sort(reverse=True)
            return os.path.join(h5_dir, files[0])
    return None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = find_model()
if not model_path:
    print("❌  No trained model found. Run train.py first.")
    sys.exit(1)

print(f"✅  Loading model: {model_path}")
verifier = MedicineVerifier(model_path)
print("✅  Server ready.\n")

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

if __name__ == "__main__":
    print("=" * 50)
    print("  MediScan — Counterfeit Medicine Detector")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)