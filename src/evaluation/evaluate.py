import pandas as pd
import yaml
import mlflow.sklearn
from sklearn.metrics import root_mean_squared_error
import os
import sys

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

def evaluate_production_model():
    print("Starting Model Evaluation...")
    
    model_name = "NYC_Taxi_Model"
    model_uri = f"models:/{model_name}/Production"
    local_model_path = "models/production_model"
    
    # Try the Registry first (Works on your laptop)
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        print(f"Attempting to load from Registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to local files (Works on GitHub Actions)
        print("Registry unreachable. Attempting to load from local artifacts...")
        if os.path.exists(local_model_path):
            model = mlflow.sklearn.load_model(local_model_path)
        else:
            print("Error: No model found in Registry or local path.")
            sys.exit(1)

    # Actual Evaluation Logic
    split_path = config["data"]["split_path"]
    X_test = pd.read_csv(os.path.join(split_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(split_path, "y_test.csv")).values.ravel()

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    
    print(f"Model RMSE: {rmse:.4f}")
    threshold = config["evaluate"]["rmse_threshold"] 
    
    if rmse < threshold:
        print(f"Validation Success: {rmse:.2f} < {threshold}")
    else:
        print(f"Validation Failure: {rmse:.2f} > {threshold}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_production_model()
