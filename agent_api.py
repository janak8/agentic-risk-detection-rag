from fastapi import APIRouter
from pydantic import BaseModel
from agent_workflow import agent_app

router = APIRouter()

class LoanApplication(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    years_employed: float
    debt_to_income: float

class UserQuestion(BaseModel):
    question: str

@router.post("/evaluate")
def evaluate_loan(application: LoanApplication):
    question = f"""
    Evaluate this loan application:
    - Age: {application.age}
    - Annual Income: {application.income}
    - Loan Amount: {application.loan_amount}
    - Credit Score: {application.credit_score}
    - Years Employed: {application.years_employed}
    - Debt-to-Income Ratio: {application.debt_to_income}
    Should this loan be approved or rejected?
    """
    result = agent_app.invoke({"question": question})
    return {
        "applicant": application.model_dump(),
        "decision": result["final_answer"]
    }

@router.post("/ask")
def ask_question(user_input: UserQuestion):
    result = agent_app.invoke({"question": user_input.question})
    return {
        "question": user_input.question,
        "answer": result["final_answer"]
    }