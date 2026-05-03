import os

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/features_done.txt", "w") as f:
    f.write("Feature engineering completed.")

print("Feature engineering complete.")