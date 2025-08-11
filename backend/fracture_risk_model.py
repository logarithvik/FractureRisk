"""
fracture_risk_model.py — isotonic calibration + prior adjustment
================================================================

What’s included (edits applied):
1) **Isotonic calibration**: fit on validation probs, saved to `artefacts/calibrator.pkl`.
2) **Return both** raw and calibrated probs at predict-time, and (optionally) a **prior-adjusted** prob
   when deployment prevalence differs from validation prevalence.
3) **Save validation prevalence** so we can adjust later.
4) Keeps your alcohol features + imputers + PR‑AUC early stopping + compact, regularized net.

Run:
  prepare → train → predict
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression

# -----------------------------
# Feature configuration
# -----------------------------
CATEGORICAL_FEATURES = ["Sex"]
NUMERICAL_FEATURES = [
    "Age",
    "BMI",
    "Smoking",
    "PastFracture",
    "Alcohol3Plus",
    "ALQ130_clean",
    "AlcoholUnknown",
]
TARGET = "Osteoporosis"

# -----------------------------
# Data preparation from NHANES SAS files
# -----------------------------

def prepare(sas_dir: str, out_csv: str):
    sas_dir = Path(sas_dir)
    demo_fp    = sas_dir / "DEMO_J.xpt"
    dxa_fp     = sas_dir / "DXX_J.xpt"
    smoking_fp = sas_dir / "SMQ_J.xpt"
    bmx_fp     = sas_dir / "BMX_J.xpt"
    osteop_fp  = sas_dir / "OSQ_J.xpt"
    alcohol_fp = sas_dir / "ALQ_J.xpt"

    demo    = pd.read_sas(demo_fp)
    dxa     = pd.read_sas(dxa_fp)
    smoking = pd.read_sas(smoking_fp)
    bmx     = pd.read_sas(bmx_fp)
    osteop  = pd.read_sas(osteop_fp)
    alcohol = pd.read_sas(alcohol_fp)

    data = (
        demo.merge(dxa, on="SEQN")
            .merge(smoking, on="SEQN", how="left")
            .merge(bmx, on="SEQN", how="left")
            .merge(osteop, on="SEQN", how="left")
            .merge(alcohol, on="SEQN", how="left")
    )

    data = data.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "SexCode",
        "BMXWT": "Weight",
        "BMXHT": "Height",
    })

    data["Sex"] = data["SexCode"].map({1: "Male", 2: "Female"})
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["Smoking"] = data["SMQ020"].map({1: 1, 2: 0}).fillna(0).astype(int)

    for col in ["OSQ020A", "OSQ020B", "OSQ020C"]:
        data[col] = data[col].fillna(0)
    data["PastFracture"] = data[["OSQ020A", "OSQ020B", "OSQ020C"]].sum(axis=1).astype(int)

    # Alcohol features from ALQ130 (1–14 exact; 15=15+; 777/999 unknown)
    alq = data.get("ALQ130")
    if alq is not None:
        alq_clean = alq.replace({777: np.nan, 999: np.nan}).astype("float")
        data["ALQ130_clean"] = alq_clean.clip(upper=15)
        data["Alcohol3Plus"] = ((alq_clean >= 3) | (alq_clean == 15)).astype(float).fillna(0.0)
        data["AlcoholUnknown"] = alq_clean.isna().astype(float)
    else:
        data["ALQ130_clean"] = np.nan
        data["Alcohol3Plus"] = 0.0
        data["AlcoholUnknown"] = 1.0

    data["Osteoporosis"] = data["OSQ060"].map({1: 1, 2: 0}).fillna(2).astype(int)

    final = data[[
        "SEQN", "Age", "Sex", "BMI", "Smoking", "PastFracture",
        "Alcohol3Plus", "ALQ130_clean", "AlcoholUnknown", "Osteoporosis",
    ]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)
    print(f"✅ Saved processed CSV to {out_csv}")
    return final

# -----------------------------
# Pre‑processing with imputers
# -----------------------------

def build_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])

# -----------------------------
# Model
# -----------------------------

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

# -----------------------------
# Helpers
# -----------------------------

def adjust_for_prior(prob: float, train_prior: float, target_prior: float) -> float:
    """Prior probability shift correction (odds scaling)."""
    eps = 1e-9
    p = float(np.clip(prob, eps, 1 - eps))
    odds = p / (1 - p)
    w_train = train_prior / max(1 - train_prior, eps)
    w_target = target_prior / max(1 - target_prior, eps)
    odds_adj = odds * (w_target / max(w_train, eps))
    return float(odds_adj / (1 + odds_adj))

# -----------------------------
# Training with PR‑AUC early stopping + isotonic calibration
# -----------------------------

def train(csv_path: str, out_dir: str, epochs: int = 50, batch_size: int = 32, no_class_weight: bool = False):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    missing = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[data[TARGET].isin([0, 1])].copy()
    X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = data[TARGET].astype(np.int32)

    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = None
    if not no_class_weight:
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

    model = build_model(input_dim=X_proc.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", patience=6, mode="max", restore_best_weights=True
    )
    rlrop = keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop, rlrop],
        verbose=2,
    )

    # ---- Evaluate & select threshold ----
    val_probs_raw = model.predict(X_val, verbose=0).ravel()
    pos_rate = float(y_val.mean())
    print(f"Validation positive rate (baseline PR‑AUC): {pos_rate:.3f}")

    prec, rec, thr = precision_recall_curve(y_val, val_probs_raw)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thr[max(best_idx - 1, 0)])

    pr_auc = float(average_precision_score(y_val, val_probs_raw))

    fpr, tpr, roc_thr = roc_curve(y_val, val_probs_raw)
    youden_j = tpr - fpr
    best_j_idx = int(np.argmax(youden_j))
    best_j_threshold = float(roc_thr[max(best_j_idx, 0)])

    y_pred = (val_probs_raw >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    # ---- Fit & save isotonic calibrator on validation set ----
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val.astype(float))
    joblib.dump(calibrator, out_dir_p / "calibrator.pkl")

    # Save artefacts
    model.save(out_dir_p / "model.keras")
    joblib.dump(preprocessor, out_dir_p / "preprocessor.pkl")

    with open(out_dir_p / "threshold.json", "w") as f:
        json.dump({
            "best_threshold_f1": best_threshold,
            "best_threshold_youdenJ": best_j_threshold,
            "pr_auc": pr_auc,
            "val_positive_rate": pos_rate,   # NEW
        }, f, indent=2)

    with open(out_dir_p / "metrics.json", "w") as f:
        json.dump({
            "history": {k: list(map(float, v)) for k, v in history.history.items()},
            "confusion_matrix_at_f1": cm.tolist(),
            "classification_report_at_f1": report,
        }, f, indent=2)

    print(f"✅ Saved model, preprocessor, calibrator, and metrics to {out_dir_p}")
    print(f"Best F1 threshold: {best_threshold:.3f} | PR-AUC: {pr_auc:.3f}")

# -----------------------------
# Inference — returns dict with raw, calibrated, and optional prior‑adjusted
# -----------------------------

def load_artefacts(model_path: str, prep_path: str):
    model = keras.models.load_model(model_path)
    pre = joblib.load(prep_path)
    return model, pre


def predict_osteoporosis(
    age: float,
    weight: float,
    height: float,
    sex: str,
    smoking: int,
    past_fracture: int,
    alcohol3plus: int = 0,
    alq130_clean: float | None = None,
    alcohol_unknown: int | None = None,
    *,
    model=None,
    preprocessor=None,
    model_path=None,
    prep_path=None,
    apply_calibration: bool = True,
    target_prior: float | None = None,
):
    """Return a dict with {prob_raw, prob, prob_prior_adjusted?}.

    - prob_raw: raw NN probability
    - prob: calibrated probability (if calibrator available and apply_calibration=True), else prob_raw
    - prob_prior_adjusted: optional prior-adjusted probability using validation prevalence
    """
    if model is None or preprocessor is None:
        if not model_path or not prep_path:
            raise ValueError("Provide model and preprocessor paths.")
        model, preprocessor = load_artefacts(model_path, prep_path)

    sex = sex.capitalize()
    if sex not in {"Male", "Female"}:
        raise ValueError("sex must be 'Male' or 'Female'")

    bmi = weight / ((height / 100) ** 2)

    if alq130_clean is None:
        alq130_clean = 3.0 if int(bool(alcohol3plus)) == 1 else 0.0
    if alcohol_unknown is None:
        alcohol_unknown = 0

    df = pd.DataFrame([[
        float(age), float(bmi), sex, int(smoking), int(past_fracture),
        int(bool(alcohol3plus)), float(alq130_clean), int(alcohol_unknown)
    ]], columns=[
        "Age", "BMI", "Sex", "Smoking", "PastFracture",
        "Alcohol3Plus", "ALQ130_clean", "AlcoholUnknown"
    ])

    X_proc = preprocessor.transform(df)
    prob_raw = float(model.predict(X_proc, verbose=0)[0][0])

    prob_cal = prob_raw
    base_dir = Path(model_path).parent if model_path else Path(prep_path).parent

    if apply_calibration:
        calib_path = base_dir / "calibrator.pkl"
        if calib_path.exists():
            try:
                calibrator = joblib.load(calib_path)
                prob_cal = float(calibrator.predict([prob_raw])[0])
            except Exception:
                prob_cal = prob_raw

    result = {"prob_raw": prob_raw, "prob": prob_cal}

    # Optional prior adjustment
    if target_prior is not None:
        try:
            meta = json.loads((base_dir / "threshold.json").read_text())
            train_prior = float(meta.get("val_positive_rate"))
            if 0 < train_prior < 1 and 0 < target_prior < 1:
                result["prob_prior_adjusted"] = adjust_for_prior(prob_cal, train_prior, target_prior)
                result["target_prior"] = target_prior
                result["train_prior"] = train_prior
        except Exception:
            pass

    return result

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Osteoporosis risk utility (calibrated & prior-adjustable)")
    sub = parser.add_subparsers(dest="mode", required=True)

    prep_p = sub.add_parser("prepare", help="Prepare CSV from SAS files")
    prep_p.add_argument("--sas_dir", default="./", help="NHANES XPT files folder")
    prep_p.add_argument("--out_csv", default="nhanes_data.csv", help="Path to save processed CSV")

    tr_p = sub.add_parser("train", help="Train osteoporosis model")
    tr_p.add_argument("--csv", default="nhanes_data.csv")
    tr_p.add_argument("--out_dir", default="artefacts")
    tr_p.add_argument("--epochs", type=int, default=50)
    tr_p.add_argument("--batch_size", type=int, default=32)
    tr_p.add_argument("--no_class_weight", action="store_true", help="Disable class weights (A/B test)")

    pd_p = sub.add_parser("predict", help="Predict osteoporosis risk")
    pd_p.add_argument("--model", default="artefacts/model.keras")
    pd_p.add_argument("--prep", default="artefacts/preprocessor.pkl")
    pd_p.add_argument("--age", type=float, required=True)
    pd_p.add_argument("--weight", type=float, required=True, help="Weight in kg")
    pd_p.add_argument("--height", type=float, required=True, help="Height in cm")
    pd_p.add_argument("--sex", choices=["Male", "Female"], required=True)
    pd_p.add_argument("--smoking", type=int, choices=[0, 1], required=True)
    pd_p.add_argument("--past_fracture", type=int, required=True, help="Count of prior fractures")
    pd_p.add_argument("--alcohol3plus", type=int, choices=[0, 1], default=0, help="1 if avg≥3 drinks on drinking days")
    pd_p.add_argument("--alq130_clean", type=float, default=None, help="Numeric ALQ130 (1–15, 15=15+")
    pd_p.add_argument("--alcohol_unknown", type=int, choices=[0, 1], default=None, help="1 if ALQ130 unknown/refused")
    pd_p.add_argument("--no_calibration", action="store_true", help="Return raw model probability (skip calibration)")
    pd_p.add_argument("--target_prior", type=float, default=None, help="Adjust calibrated prob to target prevalence (e.g., 0.15)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args.sas_dir, args.out_csv)
    elif args.mode == "train":
        train(args.csv, args.out_dir, args.epochs, args.batch_size, no_class_weight=args.no_class_weight)
    elif args.mode == "predict":
        res = predict_osteoporosis(
            age=args.age,
            weight=args.weight,
            height=args.height,
            sex=args.sex,
            smoking=args.smoking,
            past_fracture=args.past_fracture,
            alcohol3plus=args.alcohol3plus,
            alq130_clean=args.alq130_clean,
            alcohol_unknown=args.alcohol_unknown,
            model_path=args.model,
            prep_path=args.prep,
            apply_calibration=(not args.no_calibration),
            target_prior=args.target_prior,
        )
        if isinstance(res, dict):
            parts = [
                f"Raw: {res['prob_raw']:.3f}",
                f"Calibrated: {res['prob']:.3f}",
            ]
            if "prob_prior_adjusted" in res:
                parts.append(
                    f"Prior-adjusted({int(res['target_prior']*100)}%): {res['prob_prior_adjusted']:.3f}"
                )
            print(" | ".join(parts))
        else:
            print(f"Predicted osteoporosis probability: {float(res):.3f}")
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
