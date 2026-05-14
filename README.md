Risk Decision System — ML-Powered Loan Risk API
A production-style machine learning API that predicts whether a loan application is high risk or low risk based on applicant financial data. Built with a full MLOps pipeline including experiment tracking, model versioning, and containerised deployment.

 Live Demo

API Docs: http://localhost:8000/docs


How It Works
User Input → Preprocessing (StandardScaler) → Logistic Regression → Risk Decision
The pipeline automatically retrains, tracks, versions, and promotes the best model to production — all orchestrated by Prefect and tracked by MLflow.

Features

Logistic Regression model for binary risk classification
FastAPI REST API with auto-generated Swagger docs
StandardScaler preprocessing with joblib persistence
MLflow experiment tracking, model registry, and versioning
Prefect pipeline orchestration — train, evaluate, register, promote
Docker containerised deployment
Automatic production promotion — every pipeline run sets the latest model as @ production


🛠 Tech Stack
LayerTechnologyModelScikit-learn (Logistic Regression)APIFastAPI + UvicornPipelinePrefectTrackingMLflow + SQLite backendContainerisationDockerLanguagePython 3.11

Project Structure
risk_decision_system/
├── api/                  # FastAPI app and schema
│   ├── app.py
│   └── schema.py
├── modeling/             # Model class
│   └── model.py
├── preprocessing/        # Scaler/preprocessor
│   └── preprocessor.py
├── inference/            # Prediction logic
│   └── predictor.py
├── decision/             # Risk decision engine
│   └── decision_engine.py
├── pipeline/             # Prefect training pipeline
│   └── prefect_pipeline.py
├── data/                 # Training data
├── artifacts/            # Saved model and scaler
├── mlruns/               # MLflow tracking
├── Dockerfile
└── requirements.txt

Run Locally
1. Clone the repository
bashgit clone https://github.com/janak8/risk-decision-system.git
cd risk-decision-system
2. Install dependencies
bashpip install -r requirements.txt
3. Train the model
bashpython pipeline/prefect_pipeline.py
4. Run the API
bashuvicorn api.app:app --reload

Docker
Build
bashdocker build -t risk-api .
Run
bashdocker run -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
  risk-api

📡 API Reference
GET /
Health check.
json{"message": "Risk Decision System API"}
POST /predict
Request:
json{
  "age": 35,
  "income": 60000,
  "loan_amount": 15000,
  "credit_score": 650,
  "years_employed": 5,
  "debt_to_income": 0.3
}
Response:
json{
  "default_probability": 0.23,
  "risk_level": "LOW RISK",
  "loan_status": "APPROVED"
}

MLflow Tracking
Start the MLflow UI to view experiments and model versions:
bashmlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
Open http://127.0.0.1:5000

MLOps Pipeline
Every time the Prefect pipeline runs:

Loads and splits the dataset
Trains a Logistic Regression model
Evaluates accuracy
Logs params, metrics, and model to MLflow
Automatically promotes the latest version to @ production

bashpython pipeline/prefect_pipeline.py

Future Improvements

 SHAP explainability for model decisions
 JWT authentication
 CI/CD with GitHub Actions
 Deploy to AWS (ECS + ECR)
 Switch to XGBoost or ensemble model
 Add request logging and monitoring


Author
Janak Adhikari
GitHub