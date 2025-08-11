"""
fracture_risk_model.py
======================

A lightweight pipeline to prepare NHANES data, train a regularized feed-forward neural network (FNN)
with class weighting and early stopping, and predict osteoporosis risk from clinical features.

Modes:
 1. `prepare`   - Load SAS files, engineer Age, Sex, BMI, Smoking, PastFracture, Osteoporosis,
                  save as CSV.
 2. `train`     - Train model on CSV dataset with class_weight and Dropout to predict Osteoporosis.
 3. `predict`   - Run inference using clinical inputs to calculate osteoporosis probability.

Dependencies:
 - pandas, numpy, scikit-learn, tensorflow (keras), joblib
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
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# -----------------------------
# Feature configuration
# -----------------------------
CATEGORICAL_FEATURES = ["Sex"]
NUMERICAL_FEATURES = ["Age", "BMI", "Smoking", "PastFracture"]
TARGET = "Osteoporosis"

# -----------------------------
# Data preparation from NHANES SAS files
# -----------------------------

def prepare(sas_dir: str, out_csv: str):
    """
    Load NHANES XPT files, engineer features and target, and save consolidated CSV.
    Features: Age, Sex, BMI, Smoking, PastFracture (sum of fractures).
    Target: Osteoporosis (1=yes, 0=no, 2=other for unknown/NA).
    """
    sas_dir = Path(sas_dir)
    demo_fp = sas_dir / "DEMO_J.xpt"
    dxa_fp = sas_dir / "DXX_J.xpt"
    smoking_fp = sas_dir / "SMQ_J.xpt"
    bmx_fp = sas_dir / "BMX_J.xpt"
    osteop_fp = sas_dir / "OSQ_J.xpt"

    demo = pd.read_sas(demo_fp)
    dxa = pd.read_sas(dxa_fp)
    smoking = pd.read_sas(smoking_fp)
    bmx = pd.read_sas(bmx_fp)
    osteop = pd.read_sas(osteop_fp)

    data = (
        demo.merge(dxa, on="SEQN")
            .merge(smoking, on="SEQN", how="left")
            .merge(bmx, on="SEQN", how="left")
            .merge(osteop, on="SEQN", how="left")
    )

    # Rename and compute basic features
    data = data.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "SexCode",
        "BMXWT": "Weight",
        "BMXHT": "Height",
    })

    data["Sex"] = data["SexCode"].map({1: "Male", 2: "Female"})
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["Smoking"] = data["SMQ020"].map({1: 1, 2: 0}).fillna(0).astype(int)

    # Past fracture count across three questions
    for col in ["OSQ020A", "OSQ020B", "OSQ020C"]:
        data[col] = data[col].fillna(0)
    data["PastFracture"] = data[["OSQ020A", "OSQ020B", "OSQ020C"]].sum(axis=1).astype(int)

    # Osteoporosis outcome: 1=yes, 2=no in NHANES. Map to 1/0 and mark others as 2 (other/unknown)
    data["Osteoporosis"] = data["OSQ060"].map({1: 1, 2: 0}).fillna(2).astype(int)

    final = data[["SEQN", "Age", "Sex", "BMI", "Smoking", "PastFracture", "Osteoporosis"]]
    final.to_csv(out_csv, index=False)
    print(f"✅ Saved processed CSV to {out_csv}")
    return final

# -----------------------------
# Pre‑processing for modeling
# -----------------------------

def build_preprocessor():
    """Scales numerical and encodes categorical features to a dense array."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

# -----------------------------
# Model
# -----------------------------

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

# -----------------------------
# Training with class weights and early stopping + threshold selection
# -----------------------------

def train(csv_path: str, out_dir: str, epochs: int = 50, batch_size: int = 32):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    missing = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only binary labels (0/1)
    data = data[data[TARGET].isin([0, 1])].copy()

    X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = data[TARGET].astype(np.int32)

    # Filter complete rows
    mask = X.notnull().all(axis=1)
    X, y = X[mask], y[mask]

    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights from the training labels only
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = dict(zip(classes, weights))

    model = build_model(input_dim=X_proc.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=5, mode="max", restore_best_weights=True
    )
    rlrop = keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop, rlrop],
        verbose=2,
    )

    # ---------------- Evaluation & threshold selection on validation set ----------------
    val_probs = model.predict(X_val, verbose=0).ravel()

    # PR-based F1 best threshold
    prec, rec, thr = precision_recall_curve(y_val, val_probs)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thr[max(best_idx - 1, 0)])  # align sizes

    pr_auc = float(average_precision_score(y_val, val_probs))

    # ROC Youden's J (optional alt)
    fpr, tpr, roc_thr = roc_curve(y_val, val_probs)
    youden_j = tpr - fpr
    best_j_idx = int(np.argmax(youden_j))
    best_j_threshold = float(roc_thr[max(best_j_idx, 0)])

    # Metrics at best F1 threshold
    y_pred = (val_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    # Save artefacts
    model.save(out_dir_p / "model.keras")
    joblib.dump(preprocessor, out_dir_p / "preprocessor.pkl")

    with open(out_dir_p / "threshold.json", "w") as f:
        json.dump({
            "best_threshold_f1": best_threshold,
            "best_threshold_youdenJ": best_j_threshold,
            "pr_auc": pr_auc,
        }, f, indent=2)

    with open(out_dir_p / "metrics.json", "w") as f:
        json.dump({
            "history": {k: list(map(float, v)) for k, v in history.history.items()},
            "confusion_matrix_at_f1": cm.tolist(),
            "classification_report_at_f1": report,
        }, f, indent=2)

    print(f"✅ Saved model, preprocessor, and metrics to {out_dir_p}")
    print(f"Best F1 threshold: {best_threshold:.3f} | PR-AUC: {pr_auc:.3f}")

# -----------------------------
# Inference
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
    model=None,
    preprocessor=None,
    model_path=None,
    prep_path=None,
) -> float:
    """Predict osteoporosis probability given clinical inputs."""
    if model is None or preprocessor is None:
        if not model_path or not prep_path:
            raise ValueError("Provide model and preprocessor paths.")
        model, preprocessor = load_artefacts(model_path, prep_path)

    sex = sex.capitalize()
    if sex not in {"Male", "Female"}:
        raise ValueError("sex must be 'Male' or 'Female'")

    bmi = weight / ((height / 100) ** 2)
    df = pd.DataFrame(
        [[age, bmi, sex, smoking, past_fracture]],
        columns=["Age", "BMI", "Sex", "Smoking", "PastFracture"],
    )
    X_proc = preprocessor.transform(df)
    prob = float(model.predict(X_proc, verbose=0)[0][0])
    return prob

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Osteoporosis risk utility")
    sub = parser.add_subparsers(dest="mode", required=True)

    prep_p = sub.add_parser("prepare", help="Prepare CSV from SAS files")
    prep_p.add_argument("--sas_dir", default="./", help="NHANES XPT files folder")
    prep_p.add_argument("--out_csv", default="nhanes_data.csv", help="Path to save processed CSV")

    tr_p = sub.add_parser("train", help="Train osteoporosis model")
    tr_p.add_argument("--csv", default="nhanes_data.csv")
    tr_p.add_argument("--out_dir", default="artefacts")
    tr_p.add_argument("--epochs", type=int, default=50)
    tr_p.add_argument("--batch_size", type=int, default=32)

    pd_p = sub.add_parser("predict", help="Predict osteoporosis risk")
    pd_p.add_argument("--model", default="artefacts/model.keras")
    pd_p.add_argument("--prep", default="artefacts/preprocessor.pkl")
    pd_p.add_argument("--age", type=float, required=True)
    pd_p.add_argument("--weight", type=float, required=True, help="Weight in kg")
    pd_p.add_argument("--height", type=float, required=True, help="Height in cm")
    pd_p.add_argument("--sex", choices=["Male", "Female"], required=True)
    pd_p.add_argument("--smoking", type=int, choices=[0, 1], required=True)
    pd_p.add_argument("--past_fracture", type=int, required=True, help="Count of prior fractures")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args.sas_dir, args.out_csv)
    elif args.mode == "train":
        train(args.csv, args.out_dir, args.epochs, args.batch_size)
    elif args.mode == "predict":
        prob = predict_osteoporosis(
            age=args.age,
            weight=args.weight,
            height=args.height,
            sex=args.sex,
            smoking=args.smoking,
            past_fracture=args.past_fracture,
            model_path=args.model,
            prep_path=args.prep,
        )
        print(f"Predicted osteoporosis probability: {prob:.3f}")
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
