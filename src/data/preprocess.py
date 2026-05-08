import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

def build_numeric_pipeline(cfg):
    return Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["preprocess"]["numeric_imputer_strategy"])),
        ("scaler", StandardScaler()),
    ])


def build_categorical_pipeline(cfg):
    return Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["preprocess"]["categorical_imputer_strategy"])),
        ("encoder", OneHotEncoder(handle_unknown=cfg["preprocess"]["onehot_handle_unknown"])),
    ])


def build_preprocessing_pipeline(cfg):
    num_cols = cfg["features"]["numeric"]
    cat_cols = cfg["features"]["categorical"]

    preprocessor = ColumnTransformer([
        ("num", build_numeric_pipeline(cfg), num_cols),
        ("cat", build_categorical_pipeline(cfg), cat_cols),
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("select", SelectKBest(score_func=f_regression, k=cfg["selection"]["k_best"])),
    ])


def run():
    df = pd.read_csv(config["data"]["processed_path"])
    target = config["target"]

    X = df.drop(columns=[target])
    y = df[target]

    pipeline = build_preprocessing_pipeline(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    X_train = pipeline.fit_transform(X_train, y_train)
    X_test = pipeline.transform(X_test)

    split_path = config["data"]["split_path"]
    joblib.dump(pipeline, config["data"]["preprocessor_path"])
    pd.DataFrame(X_train).to_csv(f"{split_path}X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{split_path}X_test.csv", index=False)
    y_train.to_csv(f"{split_path}y_train.csv", index=False)
    y_test.to_csv(f"{split_path}y_test.csv", index=False)

    print("Preprocessing complete.")

import sys

def validate_data():
    df = pd.read_csv(config["data"]["processed_path"])
    required_cols = config["features"]["numeric"] + config["features"]["categorical"] + [config["target"]]
    for col in required_cols:
        assert col in df.columns, f"Missing mandatory column: {col}"
    assert (df[config["target"]] >= 0).all(), "Data Quality Error: Negative fare detected."
    print("✅ Data Validation Passed")

if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_data()
    else:
        run()
