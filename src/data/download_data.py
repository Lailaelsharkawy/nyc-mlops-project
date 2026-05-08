import os
import glob
import shutil
import kagglehub


def download_taxi_data():

    RAW_PATH = "data/raw/"
    os.makedirs(RAW_PATH, exist_ok=True)

    print("Fetching NYC Taxi Dataset from Kaggle...")

    path = kagglehub.dataset_download(
        "elemento/nyc-yellow-taxi-trip-data"
    )

    csv_files = glob.glob(
        os.path.join(
            path,
            "**",
            "yellow_tripdata_2016-01.csv"
        ),
        recursive=True
    )

    if not csv_files:
        raise FileNotFoundError(
            "yellow_tripdata_2016-01.csv not found."
        )

    source_file = csv_files[0]

    target_file = os.path.join(
        RAW_PATH,
        "yellow_tripdata_2016-01.csv"
    )

    shutil.copy(source_file, target_file)

    print(f"Saved to: {target_file}")


if __name__ == "__main__":
    download_taxi_data()