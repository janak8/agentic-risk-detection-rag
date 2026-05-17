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

## UI Preview

<img width="1270" height="524" alt="Image" src="https://github.com/user-attachments/assets/5290e097-4eac-432c-be94-bc46dbbb32e8" />
<img width="1256" height="467" alt="Image" src="https://github.com/user-attachments/assets/493b1f84-f8e3-4b15-91be-0a1657de9da9" />
<img width="1269" height="583" alt="Image" src="https://github.com/user-attachments/assets/1cf04b31-6ace-4ab4-a5d6-86b583dad93c" />
---

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
## MLOps Preview

<img width="1280" height="581" alt="Image" src="https://github.com/user-attachments/assets/783b992b-9263-4f13-890d-3288c7f588b5" />
<img width="1279" height="593" alt="Image" src="https://github.com/user-attachments/assets/fef15582-30b7-445c-af42-da5194fcd348" />
<img width="1280" height="593" alt="Image" src="https://github.com/user-attachments/assets/2b8c630b-6edf-4e60-9225-7cf7102d1140" />
<img width="1278" height="591" alt="Image" src="https://github.com/user-attachments/assets/8e1ff4ba-8ade-4d61-826d-66b5dbd60cda" />
<img width="824" height="221" alt="Image" src="https://github.com/user-attachments/assets/09abf87f-e76e-40ff-9e6e-106cf18d2b8b" />
<img width="1274" height="580" alt="Image" src="https://github.com/user-attachments/assets/c538897f-dc8c-48d1-a3de-9bc8b518594d" />
<img width="1268" height="588" alt="Image" src="https://github.com/user-attachments/assets/fb09030c-9229-4adf-8858-27c17135a23f" />
<img width="1278" height="597" alt="Image" src="https://github.com/user-attachments/assets/1e7051e5-2290-4f11-a55f-7d8999581213" />
<img width="1238" height="242" alt="Image" src="https://github.com/user-attachments/assets/54fa6433-1a8d-4a0a-9e34-fddc0eed9124" />
<img width="662" height="214" alt="Image" src="https://github.com/user-attachments/assets/bc757bb4-19f3-4442-9f5e-8e34903d0ac1" />
---

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
