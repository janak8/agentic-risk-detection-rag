import joblib
from sklearn.linear_model import LogisticRegression

class RiskModel:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        print("[RiskModel] Model training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1]

    def save(self, path="artifacts/model.pkl"):
        joblib.dump(self.model, path)
        print(f"[RiskModel] Model saved to {path}")

    def load(self, path="artifacts/model.pkl"):
        self.model = joblib.load(path)
        print(f"[RiskModel] Model loaded from {path}")