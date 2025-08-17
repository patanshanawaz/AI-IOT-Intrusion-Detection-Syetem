# data/sample_generate.py
import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)
# Create small synthetic dataset: 1000 rows, 20 features
n = 2000
num_features = 20
X = np.random.randn(n, num_features)
# Create 3 classes: normal, dos, probe
y = np.random.choice(["normal", "dos", "probe"], size=n, p=[0.7, 0.2, 0.1])
df = pd.DataFrame(X)
df[df.shape[1]] = y
out_path = os.path.join("data", "raw", "nsl_kdd.csv")
df.to_csv(out_path, index=False, header=False)
print("Sample dataset created at", out_path)
