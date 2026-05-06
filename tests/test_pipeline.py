import os
import importlib

def test_cleaned_file_exists():
    assert os.path.exists("data/processed/cleaned.csv")

def test_preprocessor_exists():
    assert os.path.exists("data/processed/preprocessor.pkl")

def test_splits_exist():
    assert os.path.exists("data/splits/X_train.csv")


def test_build_numeric_pipeline_has_imputer_and_scaler():
    preprocess = importlib.import_module("src.data.preprocess")
    pipeline = preprocess.build_numeric_pipeline(preprocess.config)
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps


def test_build_categorical_pipeline_has_imputer_and_encoder():
    preprocess = importlib.import_module("src.data.preprocess")
    pipeline = preprocess.build_categorical_pipeline(preprocess.config)
    assert "imputer" in pipeline.named_steps
    assert "encoder" in pipeline.named_steps


def test_build_preprocessing_pipeline_has_feature_selection():
    preprocess = importlib.import_module("src.data.preprocess")
    pipeline = preprocess.build_preprocessing_pipeline(preprocess.config)
    assert "prep" in pipeline.named_steps
    assert "select" in pipeline.named_steps