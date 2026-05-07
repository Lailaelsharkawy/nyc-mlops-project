import pandas as pd
import os

X_train = pd.read_csv("data/splits/X_train.csv")

if "trip_distance" in X_train.columns and "passenger_count" in X_train.columns:
    X_train["distance_per_passenger"] = (
        X_train["trip_distance"] /
        (X_train["passenger_count"] + 1)
    )

os.makedirs("data/processed", exist_ok=True)

X_train.to_csv(
    "data/processed/featured_train.csv",
    index=False
)

with open("data/processed/features_done.txt", "w") as f:
    f.write("Feature engineering completed successfully.")

print("Feature engineering completed successfully.")