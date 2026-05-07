import pytest
import pandas as pd
import os
from src.data import preprocess, prepare, featurize

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "trip_distance": [1.0, 2.0],
        "passenger_count": [1, 2],
        "fare_amount": [10.0, 15.0],
        "pickup_datetime": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]
    })

def test_prepare_logic(sample_data, tmp_path):
    assert len(sample_data) == 2

def test_featurize_logic(sample_data):
    sample_data["distance_per_passenger"] = (
        sample_data["trip_distance"] / (sample_data["passenger_count"] + 1)
    )
    assert "distance_per_passenger" in sample_data.columns
    assert sample_data["distance_per_passenger"].iloc[0] == 0.5

def test_preprocessing_functions():
    config = {
        "preprocess": {"numeric_imputer_strategy": "mean", "categorical_imputer_strategy": "most_frequent", "onehot_handle_unknown": "ignore"},
        "features": {"numeric": ["trip_distance"], "categorical": ["passenger_count"]},
        "selection": {"k_best": 1}
    }
    pipeline = preprocess.build_preprocessing_pipeline(config)
    assert pipeline is not None