# src/evaluate.py
import os
import numpy as np
from sklearn.metrics import classification_report
import joblib

RESULTS_DIR = "results"
PROC_DIR = os.path.join("data", "processed")

def evaluate():
    X = np.load(os.path.join(PROC_DIR, "X.npy"))
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    clf = joblib.load(os.path.join(RESULTS_DIR, "rf_model.joblib"))

    # Simple split for evaluation (or you can load a separate test set)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
