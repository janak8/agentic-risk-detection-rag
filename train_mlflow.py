import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset - fixed .read_csv typo
df = pd.read_csv("data/credit_risk_dataset.csv")

X = df.drop("default", axis=1)
y = df["default"]

# Standardized variable names (X_train instead of x_train)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Starting ML workflow
mlflow.set_experiment("risk_decision_experiment")

with mlflow.start_run():
    # Defined parameters inside the run
    C_param = 1
    max_iter_param = 200

    # Training model - moved inside the with block to ensure logging works
    model = LogisticRegression(C=C_param, max_iter=max_iter_param)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log everything
    mlflow.log_param("C", C_param)
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RiskDecisionModel"
    )

    print(f"✅ Accuracy: {accuracy}")
