import pandas as pd
import yaml

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["raw_path"])

cols = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount"
]

df = df[cols]
df = df.dropna()

df.to_csv(config["data"]["processed_path"], index=False)

print("Prepared data saved.")