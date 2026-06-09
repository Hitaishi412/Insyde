"""
features/email.py

Generate behavioral features from raw email logs.
"""
from config import FEATURE_COLUMNS
import pandas as pd


def build_email_features(df):

    df = df.copy()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Hour and weekday extraction
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday

    # External emails
    df["external"] = (
        ~df["to"].str.contains(
            "@company.com",
            case=False,
            na=False
        )
    ).astype(int)

    # Late-night emails
    df["late_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 5)
    ).astype(int)

    # Weekend emails
    df["weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    # Aggregate per user
    features = df.groupby("user").agg(

        email_count=(
            "to",
            "count"
        ),

        recipient_count=(
            "to",
            "nunique"
        ),

        external_count=(
            "external",
            "sum"
        ),

        late_night_count=(
            "late_night",
            "sum"
        ),

        weekend_count=(
            "weekend",
            "sum"
        )

    ).reset_index()

    # Ratios
    features["external_ratio"] = (
        features["external_count"]
        /
        features["email_count"]
    )

    features["late_night_ratio"] = (
        features["late_night_count"]
        /
        features["email_count"]
    )

    features["weekend_ratio"] = (
        features["weekend_count"]
        /
        features["email_count"]
    )

    # Email spike
    mean_emails = features["email_count"].mean()
    std_emails = features["email_count"].std()

    if std_emails == 0:
        std_emails = 1

    features["email_spike"] = (
        (
            features["email_count"]
            >
            mean_emails + 2 * std_emails
        )
    ).astype(int)

    return features