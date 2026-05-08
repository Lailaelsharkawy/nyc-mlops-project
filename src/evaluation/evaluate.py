import pandas as pd
import mlflow.sklearn
from sklearn.metrics import root_mean_squared_error
import yaml

def validate_model():
    with open("configs/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_df = pd.read_csv("data/splits/X_test.csv")
    y_test = pd.read_csv("data/splits/y_test.csv")

    model_uri = f"models:/TaxiFareModel/Production"
    model = mlflow.sklearn.load_model(model_uri)

    predictions = model.predict(test_df)
    rmse = root_mean_squared_error(y_test, predictions, squared=False)

    # Assert performance threshold [cite: 119-120]
    threshold = params['evaluate']['rmse_threshold']
    if rmse < threshold:
        print(f"✅ Model Validation Passed: RMSE {rmse:.2f} < {threshold}")
    else:
        raise ValueError(f"Model Validation Failed: RMSE {rmse:.2f} exceeds threshold!")

if __name__ == "__main__":
    validate_model()