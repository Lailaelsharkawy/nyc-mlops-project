import mlflow
from mlflow.tracking import MlflowClient
import yaml
import sys

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()

exp_name = config["mlflow"]["experiment_name"]
experiment = client.get_experiment_by_name(exp_name)

# --- FIX: Safety Check ---
if experiment is None:
    print(f"❌ Error: Experiment '{exp_name}' not found in {config['mlflow']['tracking_uri']}")
    available_exps = client.search_experiments()
    print("Available experiments in DB:", [e.name for e in available_exps])
    sys.exit(1)
# -------------------------

runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rmse ASC"])

if not runs:
    print(f"❌ Error: No runs found for experiment '{exp_name}'.")
    sys.exit(1)

best_run_id = runs[0].info.run_id

model_name = "NYC_Taxi_Model"
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, model_name)

# Move through stages
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging"
)

client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production"
)

print(f"✅ Success! Model version {result.version} is now in PRODUCTION.")