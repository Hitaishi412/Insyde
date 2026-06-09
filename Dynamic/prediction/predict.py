"""
prediction/predict.py

Prediction engine for Insider Threat Detection.
"""
from prediction.explain import generate_reason
from config import FEATURE_COLUMNS

import joblib
import pandas as pd
import numpy as np

# Load model artifacts once
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")





def calculate_risk_score(row):

    score = 0

    # Email
    score += row["external_ratio"] * 20
    score += row["late_night_ratio"] * 15

    # Logon
    score += row["after_hours_ratio"] * 15
    score += row["weekend_login_ratio"] * 10

    # File
    score += row["copy_ratio"] * 15
    score += row["sensitive_ratio"] * 10

    # Device
    score += row["usb_ratio"] * 10
    score += row["unknown_device_ratio"] * 5

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

    X_scaled = scaler.transform(X)

    # Isolation Forest
    anomaly = model.predict(X_scaled)

    # Convert
    df["anomaly"] = np.where(
        anomaly == -1,
        1,
        0
    )

    # Decision score
    scores = model.decision_function(X_scaled)
    anomaly_score = (
    1 -
    (
        scores - scores.min()
    )
    /
    (
        scores.max() - scores.min() + 1e-8
    )
)
    df["anomaly_score"] = (
    anomaly_score * 100
    ).round(2)

    # Risk Score
    df["risk_score"] = df.apply(
        calculate_risk_score,
        axis=1
    )

    # Boost anomalies
    df.loc[
        df["anomaly"] == 1,
        "risk_score"
    ] += 20

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
    )

    # Risk Level
    df["risk_level"] = df[
        "risk_score"
    ].apply(risk_level)
    
    df["reason"] = df.apply(
    generate_reason,
    axis=1
)
    return df