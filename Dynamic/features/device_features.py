"""
features/device_features.py

Generate behavioral features from device activity logs.
"""
from config import FEATURE_COLUMNS
import pandas as pd


def build_device_features(df):

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Time features
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday

    # After-hours usage
    df["after_hours"] = (
        (df["hour"] < 8) |
        (df["hour"] > 18)
    ).astype(int)

    # Weekend usage
    df["weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    # USB detection
    df["usb_usage"] = (
        df["device"]
        .astype(str)
        .str.lower()
        .str.contains("usb")
    ).astype(int)

    # Unknown device detection
    trusted_devices = [
        "laptop",
        "desktop",
        "workstation",
        "company_pc"
    ]

    df["unknown_device"] = (
        ~df["device"]
        .astype(str)
        .str.lower()
        .isin(trusted_devices)
    ).astype(int)

    # Aggregate
    features = df.groupby("user").agg(

        device_activity_count=(
            "device",
            "count"
        ),

        usb_insertions=(
            "usb_usage",
            "sum"
        ),

        unknown_device_count=(
            "unknown_device",
            "sum"
        ),

        after_hours_device_usage=(
            "after_hours",
            "sum"
        ),

        weekend_device_usage=(
            "weekend",
            "sum"
        )

    ).reset_index()

    # Ratios
    features["usb_ratio"] = (
        features["usb_insertions"]
        /
        features["device_activity_count"]
    )

    features["unknown_device_ratio"] = (
        features["unknown_device_count"]
        /
        features["device_activity_count"]
    )

    features["after_hours_device_ratio"] = (
        features["after_hours_device_usage"]
        /
        features["device_activity_count"]
    )

    # Device spike
    mean_activity = features[
        "device_activity_count"
    ].mean()

    std_activity = features[
        "device_activity_count"
    ].std()

    if std_activity == 0:
        std_activity = 1

    features["device_spike"] = (
        features["device_activity_count"]
        >
        mean_activity + (2 * std_activity)
    ).astype(int)

    return features
