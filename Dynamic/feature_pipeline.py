"""
features/master.py

Master feature generation pipeline.
"""

import pandas as pd

from features.email_features import build_email_features
from features.logon_features import build_logon_features
from features.file_features import build_file_features
from features.device_features import build_device_features

from schema.detector import detect_schema


# Unified feature set used by model
FEATURE_COLUMNS = [

    # Email
    "email_count",
    "recipient_count",
    "external_count",
    "late_night_count",
    "weekend_count",
    "external_ratio",
    "late_night_ratio",
    "weekend_ratio",
    "email_spike",

    # Logon
    "login_count",
    "late_logins",
    "weekend_logins",
    "after_hours_logins",
    "failed_logins",
    "late_login_ratio",
    "weekend_login_ratio",
    "after_hours_ratio",
    "login_spike",

    # File
    "file_access_count",
    "file_copy_count",
    "sensitive_file_access",
    "after_hours_file_access",
    "weekend_file_access",
    "copy_ratio",
    "sensitive_ratio",
    "after_hours_file_ratio",
    "file_spike",

    # Device
    "device_activity_count",
    "usb_insertions",
    "unknown_device_count",
    "after_hours_device_usage",
    "weekend_device_usage",
    "usb_ratio",
    "unknown_device_ratio",
    "after_hours_device_ratio",
    "device_spike"
]


def process_logs(df):

    features = []

    cols = {c.lower() for c in df.columns}

    # Email
    if {"user", "date", "to"}.issubset(cols):

        email_df = build_email_features(df)

        if email_df is not None:
            features.append(email_df)

    # Logon
    if {"user", "date", "activity"}.issubset(cols):

        logon_df = build_logon_features(df)

        if logon_df is not None:
            features.append(logon_df)

    # File
    if {"user", "date", "file_name"}.issubset(cols):

        file_df = build_file_features(df)

        if file_df is not None:
            features.append(file_df)

    # Device
    if {"user", "date", "device"}.issubset(cols):

        device_df = build_device_features(df)

        if device_df is not None:
            features.append(device_df)

    if len(features) == 0:

        # Already engineered feature dataset
        feature_cols = [
            col for col in FEATURE_COLUMNS
            if col in df.columns
        ]

        if len(feature_cols) > 10:
            return df

        raise ValueError(
            "Could not detect a supported dataset format."
        )

    master = features[0]

    for feature_df in features[1:]:

        master = master.merge(
            feature_df,
            on="user",
            how="outer"
        )

    master.fillna(0, inplace=True)

    # Ensure all model columns exist
    for col in FEATURE_COLUMNS:

        if col not in master.columns:
            master[col] = 0

    master = master[
        ["user"] + FEATURE_COLUMNS
    ]

    return master