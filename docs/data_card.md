# Data Card — NYC Taxi Dataset

## Dataset Source

NYC Taxi & Limousine Commission (TLC) Yellow Taxi dataset.

Source:
path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")

## Dataset Description

The dataset contains historical NYC taxi trip records including:
- pickup/dropoff timestamps
- trip distance
- passenger count
- fare amount
- payment type

## Schema

| Column | Description |
|---|---|
| passenger_count | Number of passengers |
| trip_distance | Distance traveled |
| fare_amount | Final trip fare |
| RateCodeID | Taxi rate type |
| payment_type | Payment method |

## Preprocessing Decisions

- Missing values removed
- Invalid fares filtered
- Feature engineering applied
- Train/test split created
- Features standardized using sklearn pipeline

## Known Biases

- Data represents NYC only
- Historical fare patterns may not generalize
- Certain neighborhoods may be underrepresented

## Privacy Notes

No direct personal identifiers are included.

## Licensing

Dataset provided publicly on kaggle.
Used for educational purposes.