import mlflow
mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db')
client = mlflow.MlflowClient()
versions = client.search_model_versions("name='RiskDecisionModel'")
latest = max(versions, key=lambda v: int(v.version))
run = client.get_run(latest.run_id)
print('artifact uri:', run.info.artifact_uri)
