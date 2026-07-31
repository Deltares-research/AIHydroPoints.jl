import os

import mlflow

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("pybridge-hello-world")

with mlflow.start_run() as run:
    mlflow.log_param("greeting", "hello")
    mlflow.log_metric("answer", 42)
    print(f"Logged run {run.info.run_id} to {tracking_uri}")
