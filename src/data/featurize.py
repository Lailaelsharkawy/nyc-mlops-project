import pandas as pd
import os

# Load train split
X_train = pd.read_csv("data/splits/X_train.csv")

# Example feature engineering
if "trip_distance" in X_train.columns and "passenger_count" in X_train.columns:
    X_train["distance_per_passenger"] = (
        X_train["trip_distance"] /
        (X_train["passenger_count"] + 1)
    )

# Create processed folder if missing
os.makedirs("data/processed", exist_ok=True)

# Save engineered data
X_train.to_csv(
    "data/processed/featured_train.csv",
    index=False
)

print("Feature engineering completed successfully.")