# backend/fracture_risk_model.py
import math
from typing import Dict, Any

# NCHS NHANES 2013–2014 FRAX-based adjusted mean 10y probabilities (Table B)
# Source: National Health Statistics Reports No.103 (2017)
FRAX_BASELINE_BY_AGE = [
    {"age_min": 40, "age_max": 49, "hip_mean_pct": 0.10, "major_mean_pct": 2.59, "note": ""},
    {"age_min": 50, "age_max": 59, "hip_mean_pct": 0.38, "major_mean_pct": 5.54, "note": ""},
    {"age_min": 60, "age_max": 69, "hip_mean_pct": 0.86, "major_mean_pct": 7.77, "note": ""},
    {"age_min": 70, "age_max": 79, "hip_mean_pct": 2.41, "major_mean_pct": 9.57, "note": "Hip mean flagged as less reliable in source table."},
    {"age_min": 80, "age_max": 200, "hip_mean_pct": None, "major_mean_pct": 11.35, "note": "Hip mean not reliable/available in source table for 80+."},
]

FRAX_BASELINE_BY_SEX = {
    "male":   {"hip_mean_pct": 0.45, "major_mean_pct": 4.38},
    "female": {"hip_mean_pct": 0.59, "major_mean_pct": 6.29},
}

def _baseline_for_age(age: int):
    for row in FRAX_BASELINE_BY_AGE:
        if row["age_min"] <= age <= row["age_max"]:
            return row
    return FRAX_BASELINE_BY_AGE[-1]

def _compare(user_pct: float, ref_pct: float):
    if ref_pct is None:
        return "No reference available"
    if user_pct >= ref_pct * 1.25:
        return "Above average"
    if user_pct <= ref_pct * 0.75:
        return "Below average"
    return "Around average"

def _risk_bucket(pct: float) -> str:
    if pct < 5:
        return "Low"
    if pct < 10:
        return "Moderate"
    if pct < 20:
        return "High"
    return "Very high"

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

    major_pct = round(major * 100, 2)
    hip_pct = round(hip * 100, 2)

    age_band = _baseline_for_age(age)
    sex_key = "female" if female == 1 else "male"
    sex_ref = FRAX_BASELINE_BY_SEX[sex_key]

    # ---- Flags / notes ----
    flags = []
    if age < 40:
        flags.append("Model is intended for adults aged 40+; interpretation below 40 may be unreliable.")
    if age >= 80 and age_band.get("hip_mean_pct") is None:
        flags.append("Population hip-risk baseline for 80+ is not available/reliable in the reference table.")

    # ---- Summary (user-friendly) ----
    age_group_label = (
        f"{age_band['age_min']}-{age_band['age_max']}"
        if age_band["age_max"] < 200 else "80+"
    )

    summary = {
        "one_liner": f"Estimated 10-year risks: Major {major_pct}%, Hip {hip_pct}%.",
        "risk_level": {
            "major": _risk_bucket(major_pct),
            "hip": _risk_bucket(hip_pct),
        },
        "comparison": {
            "age_group": age_group_label,
            "major_vs_age_mean": _compare(major_pct, age_band.get("major_mean_pct")),
            "hip_vs_age_mean": _compare(hip_pct, age_band.get("hip_mean_pct")),
            "major_vs_sex_mean": _compare(major_pct, sex_ref.get("major_mean_pct")),
            "hip_vs_sex_mean": _compare(hip_pct, sex_ref.get("hip_mean_pct")),
        },
        "context_note": (
            "The age-group and sex baselines are population averages for context and may not match your exact profile."
        ),
        "next_step_hint": (
            "If these results are concerning, consider discussing bone health screening with a clinician."
        ),
    }

    return {
        # --- keep existing fields ---
        "major_risk": major,
        "hip_risk": hip,
        "major_percent": major_pct,
        "hip_percent": hip_pct,
        "inputs_used": {
            "age": age,
            "sex": "Female" if female == 1 else "Male",
            "bmi": bmi,
            "past_fracture": past_fracture,
            "smoking": smoking,
            "alcohol3plus": alcohol3plus,
        },

        # --- new fields ---
        "summary": summary,
        "population_reference": {
            "source": "FRAX-based population means by age group (for context)",
            "selected_age_group": age_band,
            "sex_baseline": {"sex": sex_key, **sex_ref},
            "table_age_groups": FRAX_BASELINE_BY_AGE,
        },
        "flags": flags,
    }
    
