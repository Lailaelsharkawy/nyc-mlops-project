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
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    model_name = "NYC_Taxi_Model"
    model_uri = f"models:/{model_name}/Production"
    
    try:
        print(f"📡 Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error: Could not load Production model. Ensure register_model.py was run.")
        sys.exit(1)

    split_path = config["data"]["split_path"]
    X_test = pd.read_csv(os.path.join(split_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(split_path, "y_test.csv")).values.ravel()

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    
    print(f"Production Model RMSE: {rmse:.4f}")

    threshold = config["evaluate"]["rmse_threshold"] 
    
    if rmse < threshold:
        print(f"Model Validation Passed: RMSE {rmse:.2f} is below threshold {threshold}")
    else:
        print(f"Model Validation Failed: RMSE {rmse:.2f} exceeds threshold {threshold}")
        sys.exit(1) 

if __name__ == "__main__":
    evaluate_production_model()