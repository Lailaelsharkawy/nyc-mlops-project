import mlflow
from mlflow.tracking import MlflowClient
import yaml

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()

experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rmse ASC"])
best_run_id = runs[0].info.run_id

model_name = "NYC_Taxi_Model"
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, model_name)

client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production"
)

print(f"Success! Model version {result.version} is now in PRODUCTION.")