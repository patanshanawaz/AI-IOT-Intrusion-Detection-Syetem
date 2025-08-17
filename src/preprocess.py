# src/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def load_dataset():
    # Try to load train+test if available; else try single file nsl_kdd.csv
    train_path = os.path.join(RAW_DIR, "nsl_train.csv")
    test_path = os.path.join(RAW_DIR, "nsl_test.csv")
    single = os.path.join(RAW_DIR, "nsl_kdd.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"Loaded train+test: {df.shape}")
    elif os.path.exists(single):
        df = pd.read_csv(single, header=None)
        print(f"Loaded single file: {df.shape}")
    else:
        raise FileNotFoundError("Dataset not found in data/raw. Run data/sample_generate.py to create a sample or add NSL-KDD files.")

    return df

def preprocess_and_save():
    df = load_dataset()

    # Last column is label (attack/normal)
    df = df.dropna()
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).copy()

    # For object columns apply LabelEncoder, else leave numeric
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'O':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[f"col_{col}"] = le

    # Encode label: keep as-is (multiclass); also create binary label normal vs attack
    le_label = LabelEncoder()
    y_enc = le_label.fit_transform(y)
    encoders["label_encoder"] = le_label

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save processed
    proc_X_path = os.path.join(PROC_DIR, "X.npy")
    proc_y_path = os.path.join(PROC_DIR, "y.npy")
    np.save(proc_X_path, X_scaled)
    np.save(proc_y_path, y_enc)
    joblib.dump(encoders, os.path.join(PROC_DIR, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.joblib"))

    print("Preprocessing complete.")
    print(f"Saved: {proc_X_path}, {proc_y_path}, encoders and scaler.")

if __name__ == "__main__":
    preprocess_and_save()
