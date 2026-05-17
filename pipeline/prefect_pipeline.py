from prefect import flow, task
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import sqlite3
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mlflow.set_tracking_uri("sqlite:///" + os.path.join(BASE_DIR, "../mlruns/mlflow.db").replace("\\", "/"))

# Task 1: Load the data
@task
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "../data/credit_risk_dataset.csv"))
    X = df.drop("default", axis=1)
    y = df["default"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Task 2: Train the model
@task
def train_model(X_train, y_train):
    model = LogisticRegression(C=3, max_iter=300)
    model.fit(X_train, y_train)
    return model

# Task 3: Evaluate
@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Task 4: Log with MLflow
@task
def log_model(model, accuracy):
    mlflow.set_experiment("risk_decision_pipeline")

    with mlflow.start_run():
        mlflow.log_param("C", 3)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="RiskDecisionModel"
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='RiskDecisionModel'")
    latest = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias("RiskDecisionModel", "production", latest)
    print(f"✅ Version {latest} set as production. Accuracy: {accuracy}")

# Main Flow
@flow
def training_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    log_model(model, accuracy)

    print(f"✅ Pipeline complete. Accuracy: {accuracy}")


if __name__ == "__main__":
    training_pipeline()