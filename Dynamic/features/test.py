import pandas as pd

from feature_pipeline import process_logs

df = pd.read_csv(
    r"C:\Users\Lenovo\Inside Threat Detection\uploads\email.csv"
)

features = process_logs(df)

print(features.head())
