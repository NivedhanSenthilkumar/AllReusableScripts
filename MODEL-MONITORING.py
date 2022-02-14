

import mlflow

experiment_id = "some_experiment_id"

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("dropout", 0.25)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("n_layers", 5)

    mlflow.log_metric("precision", 0.76)
    mlflow.log_metric("recall", 0.92)
    mlflow.log_metric("f1", 0.83)
    mlflow.log_metric("coverage", 0.76)

