import pandas as pd
import numpy as np
import os
import logging
import time

from prometheus_client import (
    start_http_server,
    Gauge,
    Histogram,
    Counter
)

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset
)

REPORT_DIR = "src/monitoring/evidently_reports/"
LOG_DIR = "src/logs/"
LOG_FILE = os.path.join(LOG_DIR, "monitoring.log")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_VERSION = Gauge(
    'current_model_version',
    'Current version of deployed model'
)

INF_COUNT = Counter(
    'inference_count_by_status',
    'Inference count by status',
    ['status']
)

CONF_HIST = Histogram(
    'prediction_confidence_scores',
    'Prediction confidence scores'
)

DIST_HIST = Histogram(
    'feature_trip_distance_hist',
    'Feature distribution histogram'
)

PASS_HIST = Histogram(
    'feature_passenger_count_hist',
    'Feature distribution histogram'
)
DRIFT_SCORE = Gauge(
    'drift_score',
    'Detected drift percentage'
)


def simulate_production_data(df):

    drifted = df.copy()

    numeric_cols = drifted.select_dtypes(
        include=['number']
    ).columns

    if len(numeric_cols) >= 3:

        drifted[numeric_cols[0]] *= 10
        drifted[numeric_cols[1]] += 10
        drifted[numeric_cols[2]] *= 2

    return drifted

def update_metrics():

    MODEL_VERSION.set(1)

    INF_COUNT.labels(
        status='success'
    ).inc(100)

    for i in range(100):

        DIST_HIST.observe(i)
        PASS_HIST.observe(i * 2)
        CONF_HIST.observe(0.8)

def run_monitoring():

    reference_data = pd.read_csv(
        "data/processed/featured_train.csv"
    ).dropna()

    reference_data = pd.read_csv(
        "data/raw/yellow_tripdata_2015-01.csv"
    ).sample(5000)

    drifted_data = pd.read_csv(
        "data/raw/yellow_tripdata_2016-01.csv"
    ).sample(5000)
    
    common_columns = list(
        set(reference_data.columns).intersection(
            set(drifted_data.columns)
        )
    )

    reference_data = reference_data[
        common_columns
    ]

    drifted_data = drifted_data[
        common_columns
    ]

    reference_data = reference_data.dropna()

    drifted_data = drifted_data.dropna()
    
    numeric_cols = drifted_data.select_dtypes(
        include=['number']
    ).columns

    for col in numeric_cols[:3]:

        drifted_data[col] = (
            drifted_data[col] * 1.8
        )
    
    baseline_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ]
    )

    baseline_report.run(
        reference_data=reference_data,
        current_data=reference_data
    )
    baseline_report.save_html(
        os.path.join(
            REPORT_DIR,
            "baseline_report.html"
        )
    )

    drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ]
    )

    drift_report.run(
        reference_data=reference_data,
        current_data=drifted_data
    )

    drift_report.save_html(
        os.path.join(
            REPORT_DIR,
            "drift_report.html"
        )
    )

    results = drift_report.as_dict()

    drift_result = results['metrics'][0]['result']

    number_of_drifted = drift_result[
        'number_of_drifted_columns'
    ]

    total_columns = drift_result[
        'number_of_columns'
    ]

    drift_share = (
        number_of_drifted / total_columns
    )
    DRIFT_SCORE.set(drift_share)

    print(
        f"Drifted columns: "
        f"{number_of_drifted}/{total_columns}"
    )

    if drift_share > 0.20:

        warning_message = (
            f"WARNING: Drift detected on "
            f"{drift_share:.1%} of features!"
        )

        logging.warning(warning_message)

        print(warning_message)

    else:

        success_message = (
            f"No significant drift detected. "
            f"Drift share = {drift_share:.1%}"
        )

        logging.info(success_message)

        print(success_message)

if __name__ == "__main__":

    print("Starting monitoring service...")

    start_http_server(8002)

    print("Prometheus running on http://localhost:8002")

    update_metrics()

    print("Metrics updated")

    run_monitoring()

    print("Evidently reports generated")

    while True:
        time.sleep(1)
