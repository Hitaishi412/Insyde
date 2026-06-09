import pandas as pd

from features.email_features import build_email_features
from features.logon_features import build_logon_features
from features.file_features import build_file_features
from features.device_features import build_device_features

email_raw = pd.read_csv("uploads/email.csv")
logon_raw = pd.read_csv("uploads/logon.csv")
file_raw = pd.read_csv("uploads/file.csv")
device_raw = pd.read_csv("uploads/device.csv")

email_df = build_email_features(email_raw)
logon_df = build_logon_features(logon_raw)
file_df = build_file_features(file_raw)
device_df = build_device_features(device_raw)


master = email_df.merge(
    logon_df,
    on="user",
    how="outer"
)

master = master.merge(
    file_df,
    on="user",
    how="outer"
)

master = master.merge(
    device_df,
    on="user",
    how="outer"
)

master.fillna(0, inplace=True)

print(master.shape)

print(master.columns.tolist())

print(master.head())