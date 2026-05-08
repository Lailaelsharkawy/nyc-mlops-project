import os
import glob
import shutil
import kagglehub

path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")

os.makedirs("data/raw", exist_ok=True)

csv_files = glob.glob(os.path.join(path, "**", "yellow_tripdata_2015-01.csv"), recursive=True)

if not csv_files:
    raise FileNotFoundError("yellow_tripdata_2015-01.csv not found.")

source_file = csv_files[0]
target_file = "data/raw/yellow_tripdata_2015-01.csv"

shutil.copy(source_file, target_file)

print("Saved to:", target_file)