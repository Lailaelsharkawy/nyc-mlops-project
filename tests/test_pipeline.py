import os
import pandas as pd
import pytest

from src.data.prepare import prepare_data
from src.data.featurize import featurize_data
from src.data import preprocess
from src.data import download_data


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame({
        "VendorID": [1, 2],
        "passenger_count": [1, 2],
        "trip_distance": [1.0, 2.0],
        "payment_type": [1, 2],
        "fare_amount": [10.0, 15.0],
        "tip_amount": [2.0, 3.0],
        "tolls_amount": [0.0, 1.0],
        "total_amount": [12.0, 19.0]
    })


def test_prepare_data(sample_dataframe):

    os.makedirs("data/raw", exist_ok=True)

    sample_dataframe.to_csv(
        "data/raw/yellow_tripdata_2015-01.csv",
        index=False
    )

    prepare_data()

    assert os.path.exists(
        "data/processed/cleaned.csv"
    )

    cleaned = pd.read_csv(
        "data/processed/cleaned.csv"
    )

    assert len(cleaned) == 2


def test_featurize_data():

    os.makedirs("data/splits", exist_ok=True)

    df = pd.DataFrame({
        "trip_distance": [1.0, 2.0],
        "passenger_count": [1, 2]
    })

    df.to_csv(
        "data/splits/X_train.csv",
        index=False
    )

    featurize_data()

    assert os.path.exists(
        "data/processed/featured_train.csv"
    )

    featured = pd.read_csv(
        "data/processed/featured_train.csv"
    )

    assert (
        "distance_per_passenger"
        in featured.columns
    )


def test_feature_column_created():

    featured = pd.read_csv(
        "data/processed/featured_train.csv"
    )

    assert (
        "distance_per_passenger"
        in featured.columns
    )


def test_preprocessing_pipeline():

    config = {
        "preprocess": {
            "numeric_imputer_strategy": "mean",
            "categorical_imputer_strategy": "most_frequent",
            "onehot_handle_unknown": "ignore"
        },
        "features": {
            "numeric": ["trip_distance"],
            "categorical": ["passenger_count"]
        },
        "selection": {
            "k_best": 1
        }
    }

    pipeline = preprocess.build_preprocessing_pipeline(
        config
    )

    assert pipeline is not None


def test_download_module_loaded():

    assert download_data is not None


def test_preprocessing_fit_transform():

    config = {
        "preprocess": {
            "numeric_imputer_strategy": "mean",
            "categorical_imputer_strategy": "most_frequent",
            "onehot_handle_unknown": "ignore"
        },
        "features": {
            "numeric": ["trip_distance"],
            "categorical": ["payment_type"]
        },
        "selection": {
            "k_best": 1
        }
    }

    pipeline = preprocess.build_preprocessing_pipeline(
        config
    )

    df = pd.DataFrame({
        "trip_distance": [1.0, 2.0],
        "payment_type": [1, 2]
    })

    y = [10, 20]

    transformed = pipeline.fit_transform(df, y)

    assert transformed is not None


def test_download_import():

    assert hasattr(
        download_data,
        "download_taxi_data"
    )


def test_preprocess_pipeline_execution():

    config = {
        "preprocess": {
            "numeric_imputer_strategy": "mean",
            "categorical_imputer_strategy": "most_frequent",
            "onehot_handle_unknown": "ignore"
        },
        "features": {
            "numeric": [
                "trip_distance",
                "fare_amount"
            ],
            "categorical": [
                "payment_type"
            ]
        },
        "selection": {
            "k_best": 2
        }
    }

    pipeline = preprocess.build_preprocessing_pipeline(
        config
    )

    X = pd.DataFrame({
        "trip_distance": [1.0, 2.0, 3.0],
        "fare_amount": [10, 20, 30],
        "payment_type": [1, 2, 1]
    })

    y = [15, 25, 35]

    transformed = pipeline.fit_transform(
        X,
        y
    )

    assert transformed.shape[0] == 3


def test_prepare_function_exists():

    assert callable(prepare_data)


def test_featurize_function_exists():

    assert callable(featurize_data)