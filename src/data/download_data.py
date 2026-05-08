import kagglehub
import shutil
import os
os.makedirs("data/raw", exist_ok=True)

csv_files = glob.glob(os.path.join(path, "**", "yellow_tripdata_2015-01.csv"), recursive=True)

if not csv_files:
    raise FileNotFoundError("yellow_tripdata_2015-01.csv not found.")

source_file = csv_files[0]
target_file = "data/raw/yellow_tripdata_2015-01.csv"

shutil.copy(source_file, target_file)

print("Saved to:", target_file)
def download_taxi_data():
    RAW_PATH = "data/raw/"
    os.makedirs(RAW_PATH, exist_ok=True)
    
    print("Fetching NYC Taxi Dataset from Kaggle...")
    path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")
    
    filename = "yellow_tripdata_2016-01.csv"
    source_file = os.path.join(path, filename)
    destination = os.path.join(RAW_PATH, filename)
    
    if os.path.exists(source_file):
        shutil.copy(source_file, destination)
        print(f"Successfully downloaded {filename} to {RAW_PATH}")
    else:
        print(f"Error: {filename} not found. Files in download: {os.listdir(path)}")

if __name__ == "__main__":
    download_taxi_data()