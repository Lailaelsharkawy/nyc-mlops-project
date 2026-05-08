# NYC Taxi MLOps Project

This project implements an end-to-end MLOps pipeline for the NYC Yellow Taxi dataset using modern machine learning engineering practices including data versioning, experiment tracking, model serving, CI/CD, and production monitoring.

---

# Project Overview

The project predicts NYC taxi fare-related patterns using machine learning pipelines built with reproducibility, monitoring, and deployment best practices.

The system includes:

- Data versioning with DVC
- Automated preprocessing and feature engineering
- Train/test split generation
- Config-driven parameters using YAML
- MLflow experiment tracking and model registry
- Hyperparameter optimization
- FastAPI serving endpoints
- CI/CD using GitHub Actions
- Monitoring using Evidently AI, Prometheus, and Grafana
- Drift detection and runtime logging
- Unit testing with pytest

---

# Features

- DVC data versioning
- Automated preprocessing pipeline
- Train/test split generation
- Feature engineering stage
- Config-driven parameters using YAML
- Unit testing with pytest
- GitHub Actions CI/CD
- MLflow experiment tracking
- Hyperparameter optimization
- FastAPI serving
- Drift monitoring with Evidently AI
- Prometheus metrics collection
- Grafana dashboard visualization
- Data validation
- Model validation
- Runtime monitoring dashboards

---

# System Architecture

1. Raw NYC Taxi data is versioned using DVC.
2. Preprocessing and feature engineering pipelines generate training features.
3. Models are trained and tracked using MLflow.
4. Best models are registered in the MLflow Model Registry.
5. FastAPI serves predictions through REST endpoints.
6. Prometheus collects runtime metrics.
7. Grafana visualizes monitoring dashboards.
8. Evidently AI detects feature drift and data quality issues.

---

# Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core development |
| DVC | Data and pipeline versioning |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model serving |
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |
| Evidently AI | Drift detection |
| GitHub Actions | CI/CD automation |
| Pytest | Unit testing |
| Scikit-learn | Machine learning models |
| Pandera | Data validation |

---

# Project Structure

```text
NYC-MLOPS-PROJECT/
│
├── configs/                  # YAML configuration files
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   └── splits/               # Train/test splits
│
├── docs/
│   ├── model_card.md
│   ├── data_card.md
│   ├── experiment_log.csv
│   └── screenshots/
│
├── logs/                     # Deployment and monitoring logs
│
├── src/
│   ├── data/                 # Data preparation scripts
│   ├── training/             # Training and MLflow scripts
│   ├── evaluation/           # Evaluation scripts
│   ├── serving/              # FastAPI serving app
│   ├── monitoring/           # Monitoring pipeline
│   ├── grafana/              # Grafana dashboard JSON
│   └── logs/                 # Monitoring logs
│
├── tests/                    # Unit and API tests
├── .github/workflows/        # GitHub Actions CI/CD
├── dvc.yaml                  # DVC pipeline
├── dvc.lock
├── requirements.txt
└── README.md
```

---

# Quickstart

Run the project locally in 3 commands:

```bash
pip install -r requirements.txt
dvc repro
uvicorn src.serving.app:app --reload
```

Open:

- Swagger UI:
  http://127.0.0.1:8000/docs

- Prometheus:
  http://localhost:9090

- Grafana:
  http://localhost:3000

---

# Machine Learning Pipeline

The DVC pipeline contains the following stages:

1. prepare
2. preprocess
3. featurize
4. train
5. evaluate

Run the full pipeline:

```bash
dvc repro
```

Visualize the pipeline DAG:

```bash
dvc dag
```

---

# Data Pipeline

The preprocessing pipeline performs:

- missing value handling
- feature engineering
- train/test splitting
- feature scaling
- preprocessing artifact generation

Generated outputs include:

- cleaned.csv
- featured_train.csv
- preprocessor.pkl
- X_train.csv
- X_test.csv
- y_train.csv
- y_test.csv

---

# Model Training & Experiment Tracking

MLflow is used for:

- experiment tracking
- hyperparameter optimization
- metric logging
- artifact storage
- model registry
- model versioning

Tracked metrics include:

- RMSE


Launch MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

# API Serving

The serving layer is implemented using FastAPI.

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Health check |
| /predict | POST | Generate predictions |

Launch API server:

```bash
uvicorn src.serving.app:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---
# User Interface

The project includes an interactive Streamlit frontend for real-time fare prediction.

Features:
- passenger count selection
- trip distance input
- pickup time selection
- live API prediction requests
- fare estimation display

Launch the UI:

```bash
streamlit run ui.py
```

Open:

```text
http://localhost:8501
```

The UI communicates directly with the FastAPI prediction endpoint:

```text
http://127.0.0.1:8000/predict
```
---
# Monitoring Stack

The project includes production-style monitoring using:

- Evidently AI
- Prometheus
- Grafana

Monitoring features include:

- feature drift detection
- data quality monitoring
- runtime logging
- histogram monitoring
- inference metrics
- monitoring dashboards
- drift score visualization
- prediction confidence tracking

Run monitoring:

```bash
python src/monitoring/run_monitoring.py
```

Generated monitoring outputs:

- baseline_report.html
- drift_report.html
- monitoring.log

Prometheus metrics:

```text
http://localhost:8002
```

Grafana dashboard:

```text
http://localhost:3000
```

---

# CI/CD Pipeline

GitHub Actions automates:

- dependency installation
- linting using flake8
- unit testing
- coverage validation
- DVC pipeline execution
- data validation
- model validation

Workflow file:

```text
.github/workflows/ci.yml
```

---

# Testing

Run all tests:

```bash
pytest
```

Run coverage:

```bash
pytest --cov=src tests/
```

---

# Pipeline Outputs

Generated artifacts include:

- cleaned.csv
- featured_train.csv
- preprocessor.pkl
- X_train.csv
- X_test.csv
- y_train.csv
- y_test.csv
- MLflow model artifacts
- baseline_report.html
- drift_report.html
- Grafana dashboards
- Prometheus metrics
- monitoring logs

---

# Documentation

Additional project documentation:

- docs/model_card.md
- docs/data_card.md
- docs/experiment_log.csv

---

# Dataset

Dataset:
NYC TLC Yellow Taxi Dataset

Source:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Dataset includes:

- passenger_count
- trip_distance
- fare_amount
- payment_type
- RateCodeID
- pickup/dropoff timestamps

---

# Monitoring Dashboard

The Grafana dashboard visualizes:

- drift score
- prediction confidence
- inference count
- feature histograms
- monitoring uptime
- Prometheus metrics

Dashboard configuration:

```text
src/grafana/monitoring_dashboard.json
```

---

# Evidence & Screenshots

Project screenshots are stored in:

```text
docs/screenshots/
```

Including:

- MLflow runs
- Model re