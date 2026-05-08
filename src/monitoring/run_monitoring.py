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

# =========================================================
# 1. SETUP DIRECTORIES & LOGGING
# =========================================================

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

# =========================================================
# 2. PROMETHEUS METRICS
# =========================================================

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
    'Trip distance distribution'
)

PASS_HIST = Histogram(
    'feature_passenger_count_hist',
    'Passenger count distribution'
)

# =========================================================
# 3. DRIFT SIMULATION
# =========================================================

def simulate_production_data(df):
    """
    Artificially inject drift into production data.
    """

    drifted = df.copy()

    # Feature drift 1
    if 'trip_distance' in drifted.columns:
        drifted['trip_distance'] = (
            drifted['trip_distance'] * 2.5
        )

    # Feature drift 2
    if 'passenger_count' in drifted.columns:
        drifted['passenger_count'] = (
            drifted['passenger_count'] + 3
        )

    # Feature drift 3
    if 'fare_amount' in drifted.columns:
        drifted['fare_amount'] = (
            drifted['fare_amount'] * 1.2
        )

    return drifted

# =========================================================
# 4. MONITORING PIPELINE
# =========================================================

def run_monitoring():

    print("Loading datasets...")

    # Reference dataset
    reference_data = pd.read_csv(
        "data/processed/featured_train.csv"
    )

    # Current production dataset
    current_data = pd.read_csv(
        "data/splits/X_test.csv"
    )

    # Drop NaNs
    reference_data = reference_data.dropna()
    current_data = current_data.dropna()

    # Simulate drift
    drifted_data = simulate_production_data(current_data)

    # =====================================================
    # BASELINE REPORT
    # =====================================================

    print("Generating baseline report...")

    baseline_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset()        ]
    )

    baseline_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    baseline_path = os.path.join(
        REPORT_DIR,
        "baseline_report.html"
    )

    baseline_report.save(baseline_path)

    print(f"Baseline report saved to: {baseline_path}")

    # =====================================================
    # DRIFT REPORT
    # =====================================================

    print("Generating drift report...")

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

    drift_path = os.path.join(
        REPORT_DIR,
        "drift_report.html"
    )

    drift_report.save(drift_path)

    print(f"Drift report saved to: {drift_path}")

    # =====================================================
    # DRIFT ANALYSIS
    # =====================================================

    results = drift_report.as_dict()

    drift_result = results['metrics'][0]['result']

    number_of_drifted = drift_result['number_of_drifted_columns']
    total_columns = drift_result['number_of_columns']

    drift_share = (
        number_of_drifted / total_columns
    )

    print(
        f"Drifted columns: "
        f"{number_of_drifted}/{total_columns}"
    )

    # =====================================================
    # DRIFT ALERT LOGIC
    # =====================================================

    if drift_share > 0.20:

        drifted_features = []

        for col, stats in drift_result[
            'drift_by_columns'
        ].items():

            if stats['drift_detected']:
                drifted_features.append(col)

        warning_message = (
            f"WARNING: Drift detected on "
            f"{drift_share:.1%} of features! "
            f"Impacted features: {drifted_features}"
        )

        print(warning_message)

        logging.warning(warning_message)

    else:

        success_message = (
            f"No significant drift detected. "
            f"Drift share = {drift_share:.1%}"
        )

        print(success_message)

        logging.info(success_message)

    # =====================================================
    # UPDATE PROMETHEUS METRICS
    # =====================================================

    print("Updating Prometheus metrics...")

    MODEL_VERSION.set(1.0)

    INF_COUNT.labels(
        status='success'
    ).inc(len(drifted_data))

    # Histogram metrics
    if 'trip_distance' in drifted_data.columns:

        for val in drifted_data[
            'trip_distance'
        ].head(100):

            if pd.notnull(val):
                DIST_HIST.observe(float(val))

    if 'passenger_count' in drifted_data.columns:

        for val in drifted_data[
            'passenger_count'
        ].head(100):

            if pd.notnull(val):
                PASS_HIST.observe(float(val))

    # Dummy confidence values
    np.random.seed(42)

    confidence_scores = np.random.uniform(
        0.5,
        1.0,
        size=100
    )

    for score in confidence_scores:
        CONF_HIST.observe(float(score))

    logging.info(
        "Monitoring pipeline completed successfully."
    )

    print("\nMonitoring complete.")
    print(f"Reports saved in: {REPORT_DIR}")
    print(f"Logs saved in: {LOG_FILE}")

# =========================================================
# 5. MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "Starting Prometheus server on port 8000..."
    )

    start_http_server(8000)

    print(
        "Prometheus metrics available at:"
    )

    print("http://localhost:8000")

    run_monitoring()

    # Keep server alive briefly
    time.sleep(10)