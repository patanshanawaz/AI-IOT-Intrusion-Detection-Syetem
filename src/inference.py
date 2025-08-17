# src/inference.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join("results", "rf_model.joblib")
SCALER_PATH = os.path.join("data", "processed", "scaler.joblib")
ENC_PATH = os.path.join("data", "processed", "encoders.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
enc = joblib.load(ENC_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Send JSON: {"features": [v1, v2, v3, ...]}
    Features must match preprocessing (same length).
    """
    data = request.json
    features = data.get("features")
    if features is None:
        return jsonify({"error": "no features provided"}), 400

    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)
    label = enc["label_encoder"].inverse_transform(pred.astype(int))[0]
    return jsonify({"prediction": int(pred[0]), "label": str(label)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
