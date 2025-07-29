from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib

# Import helper functions directly (no “backend.” prefix)
from fracture_risk_model import predict_fracture_risk, load_artefacts

app = Flask(__name__)
CORS(app)

# Paths to saved artefacts
MODEL_PATH = Path("artefacts/model.keras")
PREP_PATH  = Path("artefacts/preprocessor.pkl")

# Load model and preprocessor at startup
try:
    model, preprocessor = load_artefacts(str(MODEL_PATH), str(PREP_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessor: {e}")

@app.route('/api/fracture-risk', methods=['POST'])
def fracture_risk():
    data = request.get_json()
    try:
        age     = float(data['age'])
        weight  = float(data['weight'])
        height  = float(data['height'])
        sex     = data['sex']
        smoking = int(data['smoking'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    try:
        # Call your helper to compute the 0–1 risk probability
        risk_prob = predict_fracture_risk(
            age=age,
            weight=weight,
            height=height,
            sex=sex,
            smoking=smoking,
            model=model,
            preprocessor=preprocessor
        )
        # Return percentage with one decimal place
        return jsonify({'risk': round(risk_prob * 100, 1)})
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
