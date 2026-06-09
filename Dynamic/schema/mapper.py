"""
mapper.py

Maps client column names to standard names.
"""
from config import FEATURE_COLUMNS
COLUMN_MAP = {

    # User columns
    "employee_id": "user",
    "userid": "user",
    "user_id": "user",
    "empid": "user",

    # Date columns
    "timestamp": "date",
    "datetime": "date",
    "time": "date",

    # Email columns
    "recipient": "to",
    "receiver": "to",
    "mail_to": "to",

    # File columns
    "filename": "file_name",

    # Device columns
    "usb_device": "device"
}


def map_columns(df):

    rename_dict = {}

    for col in df.columns:

        key = col.lower().strip()

        if key in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[key]

    return df.rename(columns=rename_dict)