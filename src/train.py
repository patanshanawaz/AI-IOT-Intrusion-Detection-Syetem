# src/train.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

PROC_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    X = np.load(os.path.join(PROC_DIR, "X.npy"))
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    return X, y

def train_and_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Save model
    joblib.dump(clf, os.path.join(RESULTS_DIR, "rf_model.joblib"))

    # Metrics
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print("Model and results saved in results/")

if __name__ == "__main__":
    train_and_save()
    os.system('/Users/shanawaz/Desktop/AI-IoT-IDS/venv/bin/python src/preprocess.py')
