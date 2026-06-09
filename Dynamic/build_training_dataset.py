import pandas as pd
import os

df = pd.read_csv( r"C:\Users\Lenovo\Inside Threat Detection\uploads\cert_cleaned.csv")

master = pd.DataFrame()

master["user"] = df["user"]

# EMAIL
master["email_count"] = df["emails_sent_per_day"]
master["recipient_count"] = df["unique_recipients_count"]
master["external_count"] = (
    df["external_emails_ratio"]
    * df["emails_sent_per_day"]
).astype(int)

master["late_night_count"] = df["emails_sent_off_hours"]
master["weekend_count"] = 0

master["external_ratio"] = df["external_emails_ratio"]
master["late_night_ratio"] = (
    df["emails_sent_off_hours"]
    /
    (df["emails_sent_per_day"] + 1)
)

master["weekend_ratio"] = 0
master["email_spike"] = (
    df["emails_sent_per_day"]
    >
    df["emails_sent_per_day"].quantile(0.95)
).astype(int)

# LOGON
master["login_count"] = (
    df["after_hours_logons"]
    +
    df["weekend_logons"]
)

master["late_logins"] = df["after_hours_logons"]
master["weekend_logins"] = df["weekend_logons"]

master["after_hours_logins"] = (
    df["after_hours_logons"]
)

master["failed_logins"] = 0

master["late_login_ratio"] = (
    df["after_hours_logons"]
    /
    (master["login_count"] + 1)
)

master["weekend_login_ratio"] = (
    df["weekend_logons"]
    /
    (master["login_count"] + 1)
)

master["after_hours_ratio"] = (
    df["after_hours_logons"]
    /
    (master["login_count"] + 1)
)

master["login_spike"] = (
    master["login_count"]
    >
    master["login_count"].quantile(0.95)
).astype(int)

# FILE
master["file_access_count"] = (
    df["total_files_copied"]
)

master["file_copy_count"] = (
    df["total_files_copied"]
)

master["sensitive_file_access"] = (
    df["filetype_pdf"]
    +
    df["filetype_doc"]
)

master["after_hours_file_access"] = (
    df["after_hours_file_copy_ratio"]
    *
    df["total_files_copied"]
)

master["weekend_file_access"] = 0

master["copy_ratio"] = 1

master["sensitive_ratio"] = (
    master["sensitive_file_access"]
    /
    (master["file_access_count"] + 1)
)

master["after_hours_file_ratio"] = (
    df["after_hours_file_copy_ratio"]
)

master["file_spike"] = (
    df["total_files_copied"]
    >
    df["total_files_copied"].quantile(0.95)
).astype(int)

# DEVICE
master["device_activity_count"] = (
    df["avg_daily_device_connections"]
)

master["usb_insertions"] = (
    df["avg_daily_device_connections"]
)

master["unknown_device_count"] = (
    df["never_used_device"]
)

master["after_hours_device_usage"] = (
    df["late_hour_device_use"]
)

master["weekend_device_usage"] = 0

master["usb_ratio"] = (
    master["usb_insertions"]
    /
    (master["device_activity_count"] + 1)
)

master["unknown_device_ratio"] = (
    master["unknown_device_count"]
    /
    (master["device_activity_count"] + 1)
)

master["after_hours_device_ratio"] = (
    master["after_hours_device_usage"]
    /
    (master["device_activity_count"] + 1)
)

master["device_spike"] = (
    master["device_activity_count"]
    >
    master["device_activity_count"].quantile(0.95)
).astype(int)

# LABEL
if "label" in df.columns:
    master["label"] = df["label"]

os.makedirs("datasets", exist_ok=True)

master.to_csv(
    "datasets/master_training.csv",
    index=False
)

print(master.shape)
print("master_training.csv created")