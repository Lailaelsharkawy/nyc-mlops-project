import os
import pandas as pd


def featurize_data():

    x_train_path = "data/splits/X_train.csv"

    if not os.path.exists(x_train_path):
        raise FileNotFoundError(
            f"File not found: {x_train_path}"
        )

    X_train = pd.read_csv(x_train_path)

    if (
        "trip_distance" in X_train.columns and
        "passenger_count" in X_train.columns
    ):

        X_train["distance_per_passenger"] = (
            X_train["trip_distance"] /
            (X_train["passenger_count"] + 1)
        )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    X_train.to_csv(
        "data/processed/featured_train.csv",
        index=False
    )

    with open(
        "data/processed/features_done.txt",
        "w"
    ) as f:

        f.write(
            "Feature engineering completed successfully."
        )

    print(
        "Feature engineering completed successfully."
    )


if __name__ == "__main__":  
    featurize_data()