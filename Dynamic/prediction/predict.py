"""
prediction/predict.py

Prediction engine for Insider Threat Detection with bulletproof risk escalation.
"""
import os
import joblib
import pandas as pd
import numpy as np
from prediction.explain import generate_reason
from config import FEATURE_COLUMNS

# --- Load model artifacts safely using absolute cloud pathing ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(DYNAMIC_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(DYNAMIC_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def calculate_risk_score(row):
    score = 0

    # 1. Base calculated values from engineered features (safe fallback defaults)
    score += row.get("external_ratio", 0) * 20
    score += row.get("late_night_ratio", 0) * 15
    score += row.get("after_hours_ratio", 0) * 15
    score += row.get("weekend_login_ratio", 0) * 10
    score += row.get("copy_ratio", 0) * 15
    score += row.get("sensitive_ratio", 0) * 10
    score += row.get("usb_ratio", 0) * 10
    score += row.get("unknown_device_ratio", 0) * 5

    # 2. Raw structural keyword overrides
    if "activity" in row and pd.notna(row["activity"]):
        act = str(row["activity"]).lower()
        if act in ["usb_copy", "file_download", "malicious_url_click"]:
            score += 45  # Instantly elevate base metric weights
        elif act in ["email", "file_write"]:
            score += 15

    return score


def analyze_threats(feature_df):
    df = feature_df.copy()

    # Ensure all columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS]

    # Run scaling transform 
    X_scaled = scaler.transform(X)

    # Isolation Forest prediction (-1 = anomaly, 1 = normal)
    anomaly = model.predict(X_scaled)
    df["anomaly"] = np.where(anomaly == -1, 1, 0)

    # Decision score normalization
    scores = model.decision_function(X_scaled)
    anomaly_score = (
        1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    )
    df["anomaly_score"] = (anomaly_score * 100).round(2)

    # --- CRITICAL RISK ASSIGNMENT ENGINE ---
    
    # Calculate baseline mathematical score
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)

    # BULLETPROOF OVERRIDE 1: If ML model flags it as an anomaly, force it to at least 75 (HIGH)
    df.loc[df["anomaly"] == 1, "risk_score"] = df["risk_score"].apply(lambda x: max(x, 75))

    # BULLETPROOF OVERRIDE 2: If the upload explicitly contains a high risk multiplier, boost it
    if "risk_multiplier" in df.columns:
        df.loc[df["risk_multiplier"] >= 2.0, "risk_score"] = df["risk_score"].apply(lambda x: max(x, 85))

    # Clean up bounds boundary clips
    df["risk_score"] = df["risk_score"].clip(0, 100)

    # Define categorical boundary buckets using the updated scores
    def assign_risk_level(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    
    # Generate human-readable logic explanations
    df["reason"] = df.apply(generate_reason, axis=1)
    
    return df
