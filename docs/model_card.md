# Model Card — NYC Taxi Fare Prediction

## Model Details

- Model Type: Random Forest Regressor
- Framework: Scikit-learn
- Tracking Tool: MLflow
- Version: v1
- Dataset: NYC Yellow Taxi Dataset

## Intended Use

The model predicts NYC taxi fare amounts based on trip information such as:
- trip distance
- passenger count
- pickup/dropoff information

The model is intended for:
- educational MLOps workflows
- fare estimation systems
- monitoring and drift detection demonstrations

## Training Data

Training data was collected from the NYC TLC Yellow Taxi dataset.

Data includes:
- passenger_count
- trip_distance
- fare_amount
- RateCodeID
- payment_type

## Evaluation Metrics

| Metric | Value |
|---|---|
| RMSE | 0.4263944991847936 |

## Subgroup Performance

Performance may vary across:
- short vs long trips
- airport rides vs local rides

## Limitations

- Historical NYC-only data
- Does not account for traffic or weather
- Limited temporal generalization
- Synthetic drift was used for monitoring experiments

## Ethical Considerations

- Model predictions may reflect historical pricing biases
- Data may under-represent certain regions or demographics
- Predictions should not be used for financial or legal decisions