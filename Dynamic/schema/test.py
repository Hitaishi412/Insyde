import pandas as pd
from config import FEATURE_COLUMNS
from detector import detect_schema
from mapper import map_columns
from validator import validate_schema

df = pd.read_csv(
    r"C:\Users\Lenovo\Inside Threat Detection\uploads\test.csv"
)

df = map_columns(df)

schema = detect_schema(df)

validate_schema(df, schema)

print("Schema:", schema)
print("Dataset Ready")