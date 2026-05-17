# Agentic Risk Detection RAG System

A production-grade ML system that combines a **Logistic Regression risk model** with a **multi-agent RAG pipeline** to evaluate loan applications. Features full MLOps infrastructure, a beautiful web UI, and Docker deployment.

---

## Live Demo

> Open `http://localhost:8000` after running Docker

---

## How It Works

```
User Input
    ↓
┌─────────────────────────────────────────┐
│           Two Evaluation Paths          │
├──────────────────┬──────────────────────┤
│  ML Prediction   │   AI Agent Pipeline  │
│  (Instant)       │   (Deep Reasoning)   │
│                  │                      │
│  StandardScaler  │  Retrieval Agent     │
│       ↓          │       ↓              │
│  Logistic        │  Reasoning Agent     │
│  Regression      │       ↓              │
│       ↓          │  Decision Agent      │
│  Risk Score      │       ↓              │
│  APPROVE/REJECT  │  APPROVE/REJECT      │
└──────────────────┴──────────────────────┘
```

---

## Features

- **ML Risk Prediction** — instant Logistic Regression prediction with hard business rules
- **Multi-Agent RAG Pipeline** — 3-agent LangGraph system: Retrieval → Reasoning → Decision
- **FAISS Vector Database** — semantic search over loan knowledge base
- **FastAPI REST API** — with auto-generated Swagger docs
- **Beautiful Web UI** — dark themed dashboard with both evaluation modes
- **MLflow Tracking** — experiment tracking, model registry, automatic production promotion
- **Prefect Orchestration** — automated training pipeline
- **Docker** — fully containerised deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Logistic Regression) |
| Agent Framework | LangGraph |
| LLM | Ollama (llama3.2) |
| Vector Database | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Pipeline | Prefect |
| Tracking | MLflow + SQLite |
| Containerisation | Docker |
| Language | Python 3.11 |

---

## Project Structure

```
agentic-risk-detection-rag/
├── api/                        # FastAPI app and schema
│   ├── app.py
│   └── schema.py
├── agent_workflow.py           # LangGraph multi-agent pipeline
├── agent_api.py                # Agent FastAPI routes
├── modeling/                   # ML model class
│   └── model.py
├── preprocessing/              # StandardScaler
│   └── preprocessor.py
├── inference/                  # Prediction logic
│   └── predictor.py
├── decision/                   # Risk decision engine
│   └── decision_engine.py
├── pipeline/                   # Prefect training pipeline
│   └── prefect_pipeline.py
├── ui/                         # Web UI
│   └── index.html
├── data/                       # Training data + knowledge base
│   ├── credit_risk_dataset.csv
│   └── knowledge.txt
├── artifacts/                  # Saved model and scaler
├── faiss_index/                # FAISS vector index
├── mlruns/                     # MLflow tracking
├── build_index.py              # FAISS index builder
├── Dockerfile
└── requirements.txt
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/janak8/agentic-risk-detection-rag.git
cd agentic-risk-detection-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

```bash
# Download from https://ollama.com
ollama pull llama3.2
ollama serve
```

### 4. Build FAISS index

```bash
python build_index.py
```

### 5. Train the ML model

```bash
python pipeline/prefect_pipeline.py
```

### 6. Run the API

```bash
uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000`

---

## Docker

### Build

```bash
docker build -t risk-api .
```

### Run

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
  risk-api
```

---

## API Reference

### `GET /` — Web UI
### `POST /predict` — ML Prediction

```json
{
  "age": 35,
  "income": 60000,
  "loan_amount": 15000,
  "credit_score": 650,
  "years_employed": 5,
  "debt_to_income": 0.3
}
```

Response:
```json
{
  "default_probability": 0.23,
  "risk_level": "LOW RISK",
  "loan_status": "APPROVED"
}
```

### `POST /evaluate` — AI Agent Evaluation

Same input as `/predict`. Returns agent reasoning and final decision.

### `POST /ask` — Ask the Agent

```json
{
  "question": "What credit score is needed to get a loan approved?"
}
```

---

## Multi-Agent Pipeline

```
Question/Application
        ↓
┌───────────────────┐
│  Retrieval Agent  │ → searches FAISS vector DB
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Reasoning Agent  │ → analyses retrieved context with LLM
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Decision Agent   │ → gives final APPROVE/REJECT verdict
└───────────────────┘
```

---

## MLOps Pipeline

Every Prefect run automatically:

1. Loads and splits the dataset
2. Trains Logistic Regression model
3. Evaluates accuracy
4. Logs to MLflow
5. Promotes latest version to `@ production`

```bash
python pipeline/prefect_pipeline.py
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Future Improvements

- [ ] Switch LLM to Groq API for cloud deployment
- [ ] AWS deployment (ECS + ECR)
- [ ] SHAP explainability
- [ ] JWT authentication
- [ ] Switch to XGBoost or ensemble model

---

## 👤 Author

**Janak Adhikari**
[GitHub](https://github.com/janak8)
