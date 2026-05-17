import mlflow.pyfunc

# Load latest model version
model = mlflow.pyfunc.load_model("models:/RiskDecisionModel/latest")

print("✅ Model loaded from MLflow")

# Example prediction
sample = [[35, 60000, 15000, 650, 5, 0.3]]

prediction = model.predict(sample)

print("Prediction:", prediction)