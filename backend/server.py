# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json

from fracture_risk_model import load_artefacts, predict_osteoporosis

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("artefacts/model.keras")
PREP_PATH  = Path("artefacts/preprocessor.pkl")
THRESH_PATH = Path("artefacts/threshold.json")

model, preprocessor = load_artefacts(str(MODEL_PATH), str(PREP_PATH))

@app.route("/api/fracture-risk", methods=["POST"])
def fracture_risk():
    data = request.get_json(force=True) or {}
    try:
        age     = float(data["age"])
        weight  = float(data["weight"])
        height  = float(data["height"])
        sex     = str(data["sex"])
        smoking = int(data["smoking"])
        past_fracture = int(data.get("past_fracture", 0))
        apply_threshold = bool(data.get("apply_threshold", False))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        # 1) Always get the probability from the model (no apply_threshold arg)
        prob = predict_osteoporosis(
            age=age, weight=weight, height=height, sex=sex,
            smoking=smoking, past_fracture=past_fracture,
            model=model, preprocessor=preprocessor
        )

        # 2) If client asked for a label, compute it here using threshold.json
        if apply_threshold:
            if not THRESH_PATH.exists():
                return jsonify({"error": "threshold.json not found; train the model first"}), 500
            with open(THRESH_PATH, "r") as f:
                th = float(json.load(f).get("best_threshold_f1", 0.5))
            return jsonify({"prob": float(prob), "label": int(prob >= th), "threshold": th})

        # 3) Otherwise return just prob
        return jsonify({"prob": float(prob)})

    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
