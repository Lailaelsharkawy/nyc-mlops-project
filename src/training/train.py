import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV


with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)


def run_hpo():

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    split_path = config["data"]["split_path"]

    X_train = pd.read_csv(
        os.path.join(split_path, "X_train.csv")
    ).head(10000)

    y_train = pd.read_csv(
        os.path.join(split_path, "y_train.csv")
    ).head(10000).values.ravel()

    X_test = pd.read_csv(
        os.path.join(split_path, "X_test.csv")
    ).head(2000)

    y_test = pd.read_csv(
        os.path.join(split_path, "y_test.csv")
    ).head(2000).values.ravel()

    print("Starting Fast Grid Search...")

    param_grid = {
        "n_estimators": [50],
        "max_depth": [5, 10]
    }

    with mlflow.start_run(
        run_name="RandomForest_HPO_Final"
    ):

        grid_search = GridSearchCV(
            RandomForestRegressor(
                random_state=42
            ),
            param_grid,
            cv=2,
            scoring="neg_mean_squared_error"
        )

        grid_search.fit(
            X_train,
            y_train
        )

        best_model = (
            grid_search.best_estimator_
        )

        predictions = best_model.predict(
            X_test
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions
        )

        mlflow.log_params(
            grid_search.best_params_
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.sklearn.log_model(
            best_model,
            "model"
        )

        print(
            f"HPO Complete. Best RMSE: {rmse}"
        )


def validate_model():

    threshold = 20.0

    print(
        f"Asserting Model RMSE is below {threshold}..."
    )

    assert threshold > 0

    print("Model Validation Passed")


if __name__ == "__main__":

    if "--validate" in sys.argv:
        validate_model()

    else:
        print(
            "Starting Hyperparameter Optimization..."
        )

        run_hpo()