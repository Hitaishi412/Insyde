"""
schema/detector.py

Automatically identifies uploaded dataset type.
"""
from config import FEATURE_COLUMNS
def detect_schema(df):

    cols = {c.lower().strip() for c in df.columns}

    # Feature dataset
    feature_cols = {
        "email_count",
        "late_night",
        "external_ratio"
    }

    if feature_cols.issubset(cols):
        return "features"

    # Email logs
    if {"user", "date", "to"}.issubset(cols):
        return "email"

    # File logs
    if {"user", "date", "file_name"}.issubset(cols):
        return "file"

    # Device logs
    if {"user", "date", "device"}.issubset(cols):
        return "device"

    # Logon logs
    if {"user", "date", "activity"}.issubset(cols):
        return "logon"

    return "unknown"