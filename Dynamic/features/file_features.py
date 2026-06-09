"""
features/file_features.py

Generate behavioral features from file activity logs.
"""
from config import FEATURE_COLUMNS
import pandas as pd


def build_file_features(df):

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Time features
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday

    # After-hours access
    df["after_hours"] = (
        (df["hour"] < 8) |
        (df["hour"] > 18)
    ).astype(int)

    # Weekend access
    df["weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    # File copy detection
    if "activity" in df.columns:

        df["file_copy"] = (
            df["activity"]
            .astype(str)
            .str.lower()
            .str.contains("copy")
        ).astype(int)

    else:
        df["file_copy"] = 0

    # Sensitive file detection
    sensitive_extensions = [
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".zip"
    ]

    df["sensitive_file"] = df["file_name"].astype(str).apply(
        lambda x: int(
            any(
                x.lower().endswith(ext)
                for ext in sensitive_extensions
            )
        )
    )

    # Aggregate
    features = df.groupby("user").agg(

        file_access_count=(
            "file_name",
            "count"
        ),

        file_copy_count=(
            "file_copy",
            "sum"
        ),

        sensitive_file_access=(
            "sensitive_file",
            "sum"
        ),

        after_hours_file_access=(
            "after_hours",
            "sum"
        ),

        weekend_file_access=(
            "weekend",
            "sum"
        )

    ).reset_index()

    # Ratios
    features["copy_ratio"] = (
        features["file_copy_count"]
        /
        features["file_access_count"]
    )

    features["sensitive_ratio"] = (
        features["sensitive_file_access"]
        /
        features["file_access_count"]
    )

    features["after_hours_file_ratio"] = (
        features["after_hours_file_access"]
        /
        features["file_access_count"]
    )

    # File spike
    mean_files = features["file_access_count"].mean()
    std_files = features["file_access_count"].std()

    if std_files == 0:
        std_files = 1

    features["file_spike"] = (
        features["file_access_count"]
        >
        mean_files + (2 * std_files)
    ).astype(int)

    return features