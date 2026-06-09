import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load training dataset
df = pd.read_csv(
    "datasets/master_training.csv"
)

print("Dataset Shape:", df.shape)

# Remove non-feature columns
drop_cols = ["user"]

if "label" in df.columns:
    drop_cols.append("label")

X = df.drop(columns=drop_cols)

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train model
model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# Create models directory
import os

os.makedirs(
    "models",
    exist_ok=True
)

# Save artifacts
joblib.dump(
    model,
    "models/model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Training Complete")
print("model.pkl saved")
print("scaler.pkl saved")