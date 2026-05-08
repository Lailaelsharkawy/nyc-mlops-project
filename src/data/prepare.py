import os
import pandas as pd
import yaml


def prepare_data():

    with open("configs/params.yaml") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}"
        )

    df = pd.read_csv(raw_path)

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

    os.makedirs(
        os.path.dirname(processed_path),
        exist_ok=True
    )

    df.to_csv(processed_path, index=False)

    print("Prepared data saved.")


if __name__ == "__main__":  
    prepare_data()