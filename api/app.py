from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from preprocessing.preprocessor import Preprocessor
from modeling.model import RiskModel
from decision.decision_engine import DecisionEngine
from inference.predictor import RiskPredictor
from api.schema import CustomerData
from agent_api import router  # import router

app = FastAPI(title="Agentic Risk Detection RAG System")

# ML Pipeline
model = RiskModel()
model.load()
preprocessor = Preprocessor()
preprocessor.load()
engine = DecisionEngine(threshold=0.6)
predictor = RiskPredictor(model, preprocessor, engine)

# Include agent routes
app.include_router(router)

# Serve UI
app.mount("/static", StaticFiles(directory="ui"), name="static")

@app.get("/")
def home():
    return FileResponse("ui/index.html")

@app.post("/predict")
def predict(customer: CustomerData):
    # Hard rules first
    if customer.credit_score < 500:
        return {"default_probability": 0.95, "risk_level": "HIGH RISK", "loan_status": "REJECTED"}
    if customer.debt_to_income > 0.6:
        return {"default_probability": 0.90, "risk_level": "HIGH RISK", "loan_status": "REJECTED"}
    if customer.loan_amount > customer.income * 5:
        return {"default_probability": 0.85, "risk_level": "HIGH RISK", "loan_status": "REJECTED"}

    # Then model prediction
    data = [
        customer.age,
        customer.income,
        customer.loan_amount,
        customer.credit_score,
        customer.years_employed,
        customer.debt_to_income
    ]
    probability, decision = predictor.predict_customer(data)
    return {
        "default_probability": float(probability),
        "risk_level": decision,
        "loan_status": "APPROVED" if decision == "LOW RISK" else "REJECTED"
    }