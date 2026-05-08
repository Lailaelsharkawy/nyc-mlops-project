import os
import glob
import shutil

try:
    import kagglehub
except ImportError:
    kagglehub = None


def download_taxi_data():

    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed."
        )

    raw_path = "data/raw/"

    os.makedirs(raw_path, exist_ok=True)

    print(
        "Fetching NYC Taxi Dataset from Kaggle..."
    )

    path = kagglehub.dataset_download(
        "elemento/nyc-yellow-taxi-trip-data"
    )

    csv_files = glob.glob(
        os.path.join(
            path,
            "**",
            "yellow_tripdata_2015-01.csv"
        ),
        recursive=True
    )

    if not csv_files:
        raise FileNotFoundError(
            "yellow_tripdata_2015-01.csv not found."
        )

    source_file = csv_files[0]

    target_file = os.path.join(
        raw_path,
        "yellow_tripdata_2015-01.csv"
    )

    shutil.copy(
        source_file,
        target_file
    )

    print(f"Saved to: {target_file}")


if __name__ == "__main__":
    download_taxi_data()