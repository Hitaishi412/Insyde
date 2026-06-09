"""
features/logon_features.py

Generate behavioral features from logon logs.
"""

import pandas as pd

from config import FEATURE_COLUMNS
def build_logon_features(df):

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Extract time information
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday

    # Late-night login
    df["late_login"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 5)
    ).astype(int)

    # Weekend login
    df["weekend_login"] = (
        df["weekday"] >= 5
    ).astype(int)

    # After-hours login
    df["after_hours"] = (
        (df["hour"] < 8) |
        (df["hour"] > 18)
    ).astype(int)

    # Failed logins (optional)
    if "activity" in df.columns:

        df["failed_login"] = (
            df["activity"]
            .astype(str)
            .str.lower()
            .str.contains("fail")
        ).astype(int)

    else:

        df["failed_login"] = 0

    # Aggregate by user
    features = df.groupby("user").agg(

        login_count=(
            "date",
            "count"
        ),

        late_logins=(
            "late_login",
            "sum"
        ),

        weekend_logins=(
            "weekend_login",
            "sum"
        ),

        after_hours_logins=(
            "after_hours",
            "sum"
        ),

        failed_logins=(
            "failed_login",
            "sum"
        )

    ).reset_index()

    # Ratios
    features["late_login_ratio"] = (
        features["late_logins"]
        /
        features["login_count"]
    )

    features["weekend_login_ratio"] = (
        features["weekend_logins"]
        /
        features["login_count"]
    )

    features["after_hours_ratio"] = (
        features["after_hours_logins"]
        /
        features["login_count"]
    )

    # Login spike
    mean_login = features["login_count"].mean()
    std_login = features["login_count"].std()

    if std_login == 0:
        std_login = 1

    features["login_spike"] = (
        features["login_count"]
        >
        mean_login + 2 * std_login
    ).astype(int)

    return features