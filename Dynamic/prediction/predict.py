"""
prediction/predict.py

Prediction engine for Insider Threat Detection.
"""
import os
import joblib
import pandas as pd
import numpy as np
from prediction.explain import generate_reason
from config import FEATURE_COLUMNS

# --- Load model artifacts safely using absolute cloud pathing ---

# 1. Get the directory path where THIS script (predict.py) lives
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Travel up one level to find the "models" folder relative to this script
# Since predict.py sits inside Dynamic/prediction/, traveling up one level gives us Dynamic/
DYNAMIC_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(DYNAMIC_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(DYNAMIC_DIR, "models", "scaler.pkl")

# 3. Load BOTH the model and the scaler using the guaranteed absolute paths
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def calculate_risk_score(row):
    score = 0

    # --- 1. SENSOR RATIO CALCULATIONS (Using safe .get fallback) ---
    score += row.get("external_ratio", 0) * 20
    score += row.get("late_night_ratio", 0) * 15

    # Logon
    score += row.get("after_hours_ratio", 0) * 15
    score += row.get("weekend_login_ratio", 0) * 10

    # File
    score += row.get("copy_ratio", 0) * 15
    score += row.get("sensitive_ratio", 0) * 10

    # Device
    score += row.get("usb_ratio", 0) * 10
    score += row.get("unknown_device_ratio", 0) * 5

    # --- 2. RAW ACTIVITY FALLBACKS ---
    # If the row has raw activity labels instead of engineered ratios, catch them directly
    if "activity" in row and pd.notna(row["activity"]):
        act = str(row["activity"]).lower()
        if act in ["usb_copy", "file_download", "malicious_url_click"]:
            score += 50  # Give it a heavy bump
        elif act in ["email", "file_write"]:
            score += 20

    # --- 3. EXPLICIT RISK MULTIPLIER BOOST ---
    if "risk_multiplier" in row and pd.notna(row["risk_multiplier"]):
        score += float(row["risk_multiplier"]) * 15

    return min(score, 100)


def risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


def analyze_threats(feature_df):
    df = feature_df.copy()

    # Ensure all columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS]

    # Run scaling transform (Now works cleanly since 'scaler' is loaded above)
    X_scaled = scaler.transform(X)

    # Isolation Forest prediction
    anomaly = model.predict(X_scaled)

    # Convert mapping (-1 for anomalies in Isolation Forest -> 1)
    df["anomaly"] = np.where(
        anomaly == -1,
        1,
        0
    )

    # Decision score normalization
    scores = model.decision_function(X_scaled)
    anomaly_score = (
        1 -
        (scores - scores.min())
        /
        (scores.max() - scores.min() + 1e-8)
    )
    df["anomaly_score"] = (anomaly_score * 100).round(2)

    # Risk Score calculation
    df["risk_score"] = df.apply(
        calculate_risk_score,
        axis=1
    )

    # Boost calculated anomalies
    df.loc[
        df["anomaly"] == 1,
        "risk_score"
    ] += 20

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
    )

    # Assign final Risk Level categorical labels
    df["risk_level"] = df["risk_score"].apply(risk_level)
    
    # Generate human-readable logic explanations
    df["reason"] = df.apply(
        generate_reason,
        axis=1
    )
    
    return df
