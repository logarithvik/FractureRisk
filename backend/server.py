from flask import Flask, request, jsonify
from flask_cors import CORS

from fracture_risk_model import frax_lite_predict

app = Flask(__name__)
CORS(app)  # allows React to call backend during local dev

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/predict")
def predict():
    payload = request.get_json(force=True) or {}

    required = ["age", "sex", "bmi"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # default missing flags to 0
    payload.setdefault("past_fracture", 0)
    payload.setdefault("smoking", 0)
    payload.setdefault("alcohol3plus", 0)

    result = frax_lite_predict(payload)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
