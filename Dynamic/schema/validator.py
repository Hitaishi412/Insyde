"""
validator.py

Validates required columns after mapping.
"""
from config import FEATURE_COLUMNS
REQUIRED_COLUMNS = {

    "email": [
        "user",
        "date",
        "to"
    ],

    "logon": [
        "user",
        "date",
        "activity"
    ],

    "file": [
        "user",
        "date",
        "file_name"
    ],

    "device": [
        "user",
        "date",
        "device"
    ],

    "features": [
        "user"
    ]
}


def validate_schema(df, schema_type):

    if schema_type not in REQUIRED_COLUMNS:

        raise ValueError(
            f"Unsupported schema type: {schema_type}"
        )

    required = REQUIRED_COLUMNS[schema_type]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return True