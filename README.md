# NYC MLOps Project

This project builds a reproducible machine learning pipeline using the NYC Yellow Taxi dataset.

## Features

* DVC data versioning
* Automated preprocessing pipeline
* Train/test split generation
* Feature engineering stage
* Config-driven parameters using YAML
* Unit testing with pytest
* GitHub Actions CI

## Project Structure

src/ → source code
data/ → raw, processed, splits
tests/ → unit tests
configs/ → parameters
monitoring/ → monitoring scripts
docs/ → documentation

## Run Locally

```bash
pip install -r requirements.txt
python download_data.py
py -m dvc repro
pytest
```

## Pipeline Stages

1. prepare
2. preprocess
3. featurize

## Outputs

* cleaned.csv
* preprocessor.pkl
* X_train.csv
* X_test.csv
* y_train.csv
* y_test.csv

## Monitoring Stack

- Evidently AI for drift monitoring
- Prometheus for metrics collection
- Grafana for dashboard visualization

Run monitoring:

python src/monitoring/run_monitoring.py

Prometheus:
http://localhost:9090

Grafana:
http://localhost:3000