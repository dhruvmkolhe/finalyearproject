# 🚀 PredictIQ - Local CLI Quickstart Guide

This guide describes how to set up, train, and run PredictIQ locally using
command-line commands.

---

## 📋 Prerequisites

Ensure you have the following installed on your machine:

- **Python 3.11+** (Make sure to check "Add Python to PATH" during
  installation) - [Download](https://www.python.org/downloads/)
- **Node.js 20+** - [Download](https://nodejs.org/)
- **Git** (optional, for pushing to GitHub) - [Download](https://git-scm.com/)

Verify software installation in your terminal:

```bash
# Check Python
python --version

# Check Node.js & npm
node --version
npm --version
```

---

## ⚙️ Step 1: Initialize Project & Environment

Navigate to the project root directory and set up your environment
configuration:

```bash
# Navigate to the project root directory
cd "c:\Users\dhruv\Downloads\final(9)\updated with rr"

# Copy the environment template to create your active configurations
copy .env.example .env

# Edit .env and enter your Groq API key
notepad .env
```

Set the `GROQ_API_KEY` inside `.env` to your actual Groq key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

> 🔑 **Get a free GROQ API key**: [Groq Console](https://console.groq.com/keys)

---

## 🐍 Step 2: Set Up Backend

Create a Python virtual environment and install the required dependencies:

```bash
# Build python virtual environment in backend/venv
python -m venv backend/venv

# Upgrade pip
backend\venv\Scripts\python -m pip install --upgrade pip

# Install required Python packages
backend\venv\Scripts\pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn openpyxl joblib sqlalchemy python-multipart slowapi python-jose[cryptography] passlib[bcrypt] pytest pytest-asyncio pytest-cov httpx
```

---

## 📦 Step 3: Set Up Frontend

Install the necessary frontend packages:

```bash
# Change directory to the frontend folder
cd frontend

# Install npm dependencies
npm install

# Return to root directory
cd ..
```

---

## 📊 Step 4: Run the Machine Learning Pipeline

Train the customer segmentation model (K-Means), scale features, and train the 5
purchase prediction classifiers (Logistic Regression, Random Forest, XGBoost,
LightGBM, and Stacking Ensemble).

Execute the pipeline runner script from the project root:

```bash
backend\venv\Scripts\python run_pipeline.py
```

> ⏱️ _This script processes the Excel raw dataset, saves intermediate features,
> fits scale parameters, and trains the ensemble model (takes ~2-3 minutes)._

---

## 🚀 Step 5: Start the Servers

You will need two separate terminal windows open (one for the backend API and
one for the frontend web app).

### Terminal 1: Backend API Server

Navigate to the project root directory and launch the FastAPI server:

```bash
backend\venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- **API Health endpoint**:
  [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive Documentation**:
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Terminal 2: Frontend Web Application

Open a new terminal window, navigate to the frontend folder, and launch the dev
environment:

```bash
cd frontend
npm run dev
```

- **Local Application Link**: [http://localhost:5177](http://localhost:5177)

---

## 🎯 Verification & Testing

Verify that the platform is up and responding.

### 1. Test Backend API Response

Open a new terminal and inspect the health check details:

```bash
curl http://127.0.0.1:8000/api/health
```

**Expected Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "models_loaded": {
      "logistic_regression": true,
      "random_forest": true,
      "xgboost": true,
      "lightgbm": true,
      "stacking_ensemble": true,
      "scaler": true
    }
  }
}
```

### 2. Verify Frontend Pages

Open [http://localhost:5177](http://localhost:5177) in your browser:

- **Dashboard**: Review summary stats and active purchase trends.
- **Segments**: Explore segment clusters (Champions, Loyal, At-Risk, Lost
  Customers).
- **Models**: Compare training metrics, validation results, and ROC-AUC curves.
- **Predict**: Submit single inputs or upload CSV files for batch predictions.
- **Live**: Start streaming real-time simulation predictions.
- **Drift**: Monitor data stability using the Population Stability Index.
- **Chat**: Talk with the specialized retail business assistant.

---

## 🛠️ Common Troubleshooting

- **ModuleNotFoundError: No module named 'backend'** Ensure you are running the
  backend uvicorn command from the **project root directory** using
  `backend\venv\Scripts\python -m uvicorn backend.main:app` so Python is aware
  of package boundaries.
- **Port 8000 is already in use** Run:
  ```cmd
  netstat -ano | findstr :8000
  taskkill /PID <PID_NUMBER> /F
  ```
  Or change the port in the uvicorn start command: `--port 8001`.
- **Database operational connection error** Ensure the backend directory
  contains `database.db`. If corrupted, delete it with
  `del backend\database.db`, and restart the FastAPI server (it will
  automatically bootstrap and rebuild clean tables).
