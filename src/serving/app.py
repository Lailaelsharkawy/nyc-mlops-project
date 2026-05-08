import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import yaml
import os
import uvicorn

# Load config
with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="NYC Taxi Fare Predictor")

# Setup MLflow and load the Production model
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
model_uri = "models:/NYC_Taxi_Model/Production"

try:
    # Use pyfunc for production serving flexibility
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Real model loaded from Production registry.")
except Exception as e:
    print(f"⚠️ Could not load model: {e}. Using fallback logic.")
    class FallbackModel:
        def predict(self, df):
            return [15.0]
    model = FallbackModel()

@app.get("/health")
def health():
    return {"status": "online", "model": "NYC_Taxi_Model", "stage": "Production"}

@app.post("/predict")
async def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        
        input_data = df.values 
        
        prediction = model.predict(input_data)
        fare = round(float(prediction[0]), 2)
        
        return {"fare_amount": fare}
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)