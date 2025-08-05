"""
fracture_risk_model.py
======================

A lightweight pipeline to prepare NHANES data, train a regularized feed-forward neural network (FNN) with class weighting and early stopping,
…and predict osteoporosis risk from clinical features, including past fracture history.

Modes:
 1. `prepare`   - Load SAS files, engineer Age, Sex, BMI, Smoking, PastFracture, Osteoporosis,
                  save as CSV.
 2. `train`     - Train model on CSV dataset with class_weight and Dropout to predict Osteoporosis.
 3. `predict`   - Run inference using clinical inputs to calculate osteoporosis probability.

Dependencies:
 - pandas, numpy, scikit-learn, keras, joblib
"""

import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

# Use standalone Keras
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.callbacks import EarlyStopping

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# Feature configuration
CATEGORICAL_FEATURES = ["Sex"]
NUMERICAL_FEATURES = ["Age", "BMI", "Smoking", "PastFracture"]
TARGET = "Osteoporosis"

# ----------------------------------------------------------------------
# Data preparation from NHANES SAS files
# ----------------------------------------------------------------------
def prepare(sas_dir: str, out_csv: str):
    """
    Load NHANES XPT files, engineer features and target, and save consolidated CSV.
    Features: Age, Sex, BMI, Smoking, PastFracture (sum of fractures).
    Target: Osteoporosis (1=yes, 0=no, 2=other).
    """
    demo_fp = Path(sas_dir) / "DEMO_J.xpt"
    dxa_fp = Path(sas_dir) / "DXX_J.xpt"
    smoking_fp = Path(sas_dir) / "SMQ_J.xpt"
    bmx_fp = Path(sas_dir) / "BMX_J.xpt"
    osteop_fp = Path(sas_dir) / "OSQ_J.xpt"

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
        'RIDAGEYR': 'Age',
        'RIAGENDR': 'SexCode',
        'BMXWT': 'Weight',
        'BMXHT': 'Height'
    })
    data['Sex'] = data['SexCode'].map({1: 'Male', 2: 'Female'})
    data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
    data['Smoking'] = data['SMQ020'].map({1: 1, 2: 0}).fillna(0).astype(int)

    # Past fracture count across three questions
    for col in ['OSQ020A', 'OSQ020B', 'OSQ020C']:
        data[col] = data[col].fillna(0)
    data['PastFracture'] = data[['OSQ020A','OSQ020B','OSQ020C']].sum(axis=1).astype(int)

    # Osteoporosis outcome: 1=yes, 0=no, 2=other
    data['Osteoporosis'] = data['OSQ060'].map({1:1, 2:0}).fillna(2).astype(int)

    final = data[['SEQN', 'Age', 'Sex', 'BMI', 'Smoking', 'PastFracture', 'Osteoporosis']]
    final.to_csv(out_csv, index=False)
    print(f"✅ Saved processed CSV to {out_csv}")
    return final

# ----------------------------------------------------------------------
# Pre‑processing for modeling
# ----------------------------------------------------------------------
def build_preprocessor():
    """Scales numerical and encodes categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ], remainder="drop",)

# ----------------------------------------------------------------------
# Model definition with Dropout
# ----------------------------------------------------------------------
def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name="auc")],
    )
    return model

# ----------------------------------------------------------------------
# Training with class weights and early stopping
# ----------------------------------------------------------------------
def train(csv_path: str, out_dir: str, epochs: int=50, batch_size: int=32):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    missing = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = data[TARGET]

    # Filter complete rows
    mask = X.notnull().all(axis=1)
    X, y = X[mask], y[mask]

    # Class weights for imbalance
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(zip(np.unique(y), weights))

    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y)

    # Oversample minority
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = build_model(input_dim=X_proc.shape[1])
    early_stop = EarlyStopping(
        monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights, callbacks=[early_stop], verbose=2)

    artefacts_dir = out_dir
    model.save(artefacts_dir / "model.keras")
    joblib.dump(preprocessor, artefacts_dir / "preprocessor.pkl")
    print(f"✅ Saved model and preprocessor to {artefacts_dir}")

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def load_artefacts(model_path: str, prep_path: str):
    return load_model(model_path), joblib.load(prep_path)

def predict_osteoporosis(age: float, weight: float, height: float, sex: str, smoking: int, past_fracture: int,
                         model=None, preprocessor=None, model_path=None, prep_path=None) -> float:
    """
    Predict osteoporosis probability given clinical inputs.
    """
    if model is None or preprocessor is None:
        if not model_path or not prep_path:
            raise ValueError("Provide model and preprocessor paths.")
        model, preprocessor = load_artefacts(model_path, prep_path)

    sex = sex.capitalize()
    if sex not in {'Male','Female'}:
        raise ValueError("sex must be 'Male' or 'Female'")
    bmi = weight / ((height / 100) ** 2)
    df = pd.DataFrame([[age, bmi, sex, smoking, past_fracture]],
                      columns=['Age','BMI','Sex','Smoking','PastFracture'])
    X_proc = preprocessor.transform(df)
    return float(model.predict(X_proc, verbose=0)[0][0])

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
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
    pd_p.add_argument("--sex", choices=["Male","Female"], required=True)
    pd_p.add_argument("--smoking", type=int, choices=[0,1], required=True)
    pd_p.add_argument("--past_fracture", type=int, required=True, help="Count of prior fractures")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'prepare':
        prepare(args.sas_dir, args.out_csv)
    elif args.mode == 'train':
        train(args.csv, args.out_dir, args.epochs, args.batch_size)
    elif args.mode == 'predict':
        prob = predict_osteoporosis(
            age=args.age, weight=args.weight, height=args.height,
            sex=args.sex, smoking=args.smoking, past_fracture=args.past_fracture,
            model_path=args.model, prep_path=args.prep)
        print(f"Predicted osteoporosis probability: {prob:.3f}")
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
