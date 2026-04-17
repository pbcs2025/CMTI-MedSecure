"""
Flask backend for two-stage counterfeit medicine detection + barcode validation.
"""

import json
import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

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

def find_models():
    stage1 = "model/stage1_medicine_detector.h5"
    stage2 = "model/stage2_authenticity_classifier.h5"
    # Backward compatible fallback to previous single-model setup.
    legacy = "model/counterfeit_mobilenetv2_e30_b32.h5"
    if os.path.exists(stage2):
        return stage1 if os.path.exists(stage1) else None, stage2
    if os.path.exists(legacy):
        return None, legacy
    return None, None

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GENUINE_DB_PATH = "genuine_db.json"
FAKE_DB_PATH = "fake_db.json"

stage1_model, stage2_model = find_models()
if not stage2_model:
    print("[ERROR] No model found. Train Stage 2 first with: python train.py")
    sys.exit(1)

if stage1_model:
    print(f"[OK] Loading Stage 1 model: {stage1_model}")
else:
    print("[WARN] Stage 1 model missing. Running in Stage-2-only compatibility mode.")
print(f"[OK] Loading Stage 2 model: {stage2_model}")
verifier = MedicineVerifier(stage1_model, stage2_model)
print("[OK] Server ready.\n")


def _load_json_db(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


GENUINE_DB = _load_json_db(GENUINE_DB_PATH)
FAKE_DB = _load_json_db(FAKE_DB_PATH)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def lookup_barcode(barcode: str):
    if barcode in GENUINE_DB:
        return {
            "status": "VALID",
            "barcode": barcode,
            "data": GENUINE_DB[barcode],
        }
    if barcode in FAKE_DB:
        reason = FAKE_DB[barcode].get("reason", "Known counterfeit barcode")
        return {
            "status": "FAKE",
            "barcode": barcode,
            "message": f"⚠ Fake medicine detected: {reason}",
            "data": FAKE_DB[barcode],
        }
    return {
        "status": "NOT FOUND",
        "barcode": barcode,
        "message": "Medicine not found in database",
    }


def combine_results(image_result: dict, barcode_result: dict):
    image_status = image_result.get("status")
    barcode_status = barcode_result.get("status")
    image_verdict = "GENUINE" if image_status == "genuine" else "FAKE"
    barcode_verdict = "GENUINE" if barcode_status == "VALID" else "FAKE"

    if barcode_verdict == "GENUINE" and image_verdict == "GENUINE":
        final_status = "genuine"
        final_label = "Genuine"
        final_message = (
            "Barcode is genuine and image analysis is genuine. "
            "Final verdict: medicine appears genuine."
        )
        final_risk = "low"
    elif barcode_verdict == "GENUINE" and image_verdict == "FAKE":
        final_status = "suspicious"
        final_label = "Suspicious"
        final_message = (
            "Barcode verification is genuine, but image analysis is suspicious/fake. "
            "Final verdict: medicine is suspected as fake."
        )
        final_risk = "high"
    elif barcode_verdict == "FAKE" and image_verdict == "GENUINE":
        final_status = "suspicious"
        final_label = "Suspicious"
        final_message = (
            "Image analysis is genuine, but barcode is fake or not found in trusted records. "
            "Final verdict: medicine is suspected as fake."
        )
        final_risk = "high" if barcode_status == "FAKE" else "moderate"
    else:
        final_status = "suspicious"
        final_label = "Suspicious"
        final_message = (
            "Both barcode verification and image analysis indicate suspicious/fake signals. "
            "Final verdict: medicine is suspected as fake."
        )
        final_risk = "high" if image_status == "suspicious" else "moderate"

    return {
        "status": final_status,
        "label": final_label,
        "message": final_message,
        "risk_level": final_risk,
        "confidence": round(image_result.get("confidence", 0.0) * 100, 1),
        "raw_score": image_result.get("raw_score", 0.0),
        "mode": "barcode_plus_image",
        "image_result": image_result,
        "barcode_result": barcode_result,
        "component_analysis": {
            "barcode_verdict": barcode_verdict,
            "barcode_status": barcode_status,
            "image_verdict": image_verdict,
            "image_status": image_status,
        },
    }

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
        "message":    result["message"],
        "mode": "image_only",
    })


@app.route("/get-medicine", methods=["POST"])
def get_medicine():
    data = request.get_json(silent=True) or {}
    barcode = str(data.get("barcode", "")).strip()
    if not barcode:
        return jsonify({"error": "barcode is required"}), 400
    return jsonify(lookup_barcode(barcode))


@app.route("/predict-with-barcode", methods=["POST"])
def predict_with_barcode():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    barcode = str(request.form.get("barcode", "")).strip()
    if not barcode:
        return jsonify({"error": "barcode is required"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use JPG, PNG or WEBP"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        image_result = verifier.predict(filepath)
        barcode_result = lookup_barcode(barcode)
        combined = combine_results(image_result, barcode_result)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": str(e)}), 500

    os.remove(filepath)
    return jsonify(combined)

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            "genuine_db_count": len(GENUINE_DB),
            "fake_db_count": len(FAKE_DB),
        }
    )


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