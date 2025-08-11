# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json

from fracture_risk_model import load_artefacts, predict_osteoporosis

app = Flask(__name__)
CORS(app)

MODEL_PATH   = Path("artefacts/model.keras")
PREP_PATH    = Path("artefacts/preprocessor.pkl")
THRESH_PATH  = Path("artefacts/threshold.json")

# Blending weight (can be tuned; consider saving/reading from a config file later)
BLEND_ALPHA = 0.5  # final = alpha * prior_adjusted + (1-alpha) * raw

# Load model + preprocessor once
model, preprocessor = load_artefacts(str(MODEL_PATH), str(PREP_PATH))

@app.route("/api/fracture-risk", methods=["POST"])
def fracture_risk():
    data = request.get_json(force=True) or {}
    try:
        age            = float(data["age"])
        weight         = float(data["weight"])
        height         = float(data["height"])
        sex            = str(data["sex"])
        smoking        = int(data["smoking"])
        past_fracture  = int(data.get("past_fracture", 0))
        apply_threshold = bool(data.get("apply_threshold", False))

        # Optional extras
        alcohol3plus   = int(data.get("alcohol3plus", 0))
        target_prior   = data.get("target_prior", None)
        if target_prior is not None:
            target_prior = float(target_prior)
            if not (0.0 < target_prior < 1.0):
                target_prior = None
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        # Call model: return dict with prob_raw/prob and maybe prob_prior_adjusted
        res = predict_osteoporosis(
            age=age,
            weight=weight,
            height=height,
            sex=sex,
            smoking=smoking,
            past_fracture=past_fracture,
            alcohol3plus=alcohol3plus,
            model=model,
            preprocessor=preprocessor,
            apply_calibration=True,
            target_prior=target_prior,
        )

        # Normalize possible return types
        if isinstance(res, dict):
            prob_raw   = float(res.get("prob_raw", res.get("prob", 0.0)))
            prob_cal   = float(res.get("prob", prob_raw))
            prob_prior = float(res.get("prob_prior_adjusted", prob_cal))
        else:
            # Back-compat: model returned a single float
            prob_raw = float(res)
            prob_cal = prob_raw
            prob_prior = prob_raw

        # Blended probability (blend after prior adjustment)
        prob_blended = BLEND_ALPHA * prob_prior + (1.0 - BLEND_ALPHA) * prob_raw

        # If client asked for a label, use saved F1 threshold
        out = {
            "prob_raw": prob_raw,
            "prob": prob_cal,
            "prob_prior_adjusted": prob_prior,
            "prob_blended": prob_blended,
        }

        if apply_threshold:
            if not THRESH_PATH.exists():
                return jsonify({"error": "threshold.json not found; train the model first"}), 500
            with open(THRESH_PATH, "r") as f:
                meta = json.load(f)
            th = float(meta.get("best_threshold_f1", 0.5))
            out.update({
                "label": int(prob_blended >= th),  # classify using blended prob
                "threshold": th,
            })

        return jsonify(out)

    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500


if __name__ == "__main__":
    # Restart the server after editing this file
    app.run(host="0.0.0.0", port=5000, debug=True)
