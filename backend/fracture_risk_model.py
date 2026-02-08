# backend/fracture_risk_model.py
import math
from typing import Dict, Any

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def frax_lite_predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    FRAX-lite (no BMD, no RA/steroids/parent hip/sec osteo).
    Uses: Age, Sex, BMI, PastFracture, Smoking, Alcohol3Plus
    Returns: major_risk, hip_risk + percentages
    """
    age = int(inputs.get("age"))
    sex = str(inputs.get("sex", "")).strip().lower()
    bmi = float(inputs.get("bmi"))

    past_fracture = int(inputs.get("past_fracture", 0))
    smoking = int(inputs.get("smoking", 0))
    alcohol3plus = int(inputs.get("alcohol3plus", 0))

    female = 1 if sex.startswith("f") else 0

    # MOF (Major osteoporotic fracture) — FRAX-lite coefficients
    lp_major = (
        -4.251
        + 0.0133 * age
        + 0.358 * female
        + 0.00045 * bmi
        + 0.912 * past_fracture
        + 0.271 * smoking
        + 0.188 * alcohol3plus
    )
    major = _sigmoid(lp_major)

    # Hip — FRAX-lite coefficients
    lp_hip = (
        -10.763
        + 0.0753 * age
        - 0.0377 * female
        + 0.0198 * bmi
        + 0.844 * past_fracture
        + 0.801 * smoking
        + 0.308 * alcohol3plus
    )
    hip = _sigmoid(lp_hip)

    return {
        "major_risk": major,
        "hip_risk": hip,
        "major_percent": round(major * 100, 2),
        "hip_percent": round(hip * 100, 2),
        "inputs_used": {
            "age": age,
            "sex": "Female" if female == 1 else "Male",
            "bmi": bmi,
            "past_fracture": past_fracture,
            "smoking": smoking,
            "alcohol3plus": alcohol3plus,
        },
    }
