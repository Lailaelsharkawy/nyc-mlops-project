import pandas as pd
from fastapi import FastAPI, HTTPException
import logging
import os
import uvicorn
import os
import logging

current_file_path = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_file_path, "..", "logs")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filepath = os.path.join(log_dir, 'deployment.log')

logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

app = FastAPI(title="NYC Taxi Fare Predictor")

import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/NYC_Taxi_Model/Production"
)

logging.info("API initialized successfully.")

@app.get("/health")
def health():
    return {"status": "online"}

@app.post("/predict")
async def predict(data: dict):
    try:
        logging.info(f"Input Data Received: {data}")
        
        df = pd.DataFrame([data])
        
        prediction = model.predict(df)
        fare = round(float(prediction[0]), 2)
        
        logging.info(f"Prediction successful: ${fare}")
        return {"fare_amount": fare}
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid data format")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)