import os

def test_cleaned_file_exists():
    assert os.path.exists("data/processed/cleaned.csv")

def test_preprocessor_exists():
    assert os.path.exists("data/processed/preprocessor.pkl")

def test_splits_exist():
    assert os.path.exists("data/splits/X_train.csv")