from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib
from keras.models import load_model

# Import prediction function from your module
from backend.fracture_risk_model import predict_fracture_risk, load_artefacts

app = Flask(__name__)
CORS(app)

# Paths to saved artefacts
MODEL_PATH = Path("artefacts/model.keras")
PREP_PATH = Path("artefacts/preprocessor.pkl")

# Load model and preprocessor at startup
model, preprocessor = load_artefacts(str(MODEL_PATH), str(PREP_PATH))

@app.route('/api/fracture-risk', methods=['POST'])
def fracture_risk():
    data = request.get_json()
    try:
        age    = float(data['age'])
        weight = float(data['weight'])
        height = float(data['height'])
        sex    = data['sex']
        smoking= int(data['smoking'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input'}), 400

    # Compute risk (0-1)
    prob = predict_fracture_risk(
        age=age, weight=weight, height=height,
        sex=sex, smoking=smoking,
        model=model, preprocessor=preprocessor
    )
    # Return as percentage
    return jsonify({'risk': round(prob * 100, 1)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
