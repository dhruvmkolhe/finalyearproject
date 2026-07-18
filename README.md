# PredictIQ - Enterprise Customer Segmentation & Purchase Prediction Platform

[![CI/CD Pipeline](https://github.com/yourusername/predictiq/workflows/PredictIQ%20CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/predictiq/actions)
[![codecov](https://codecov.io/gh/yourusername/predictiq/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/predictiq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

> **Production-ready ML system for real-time customer behavior prediction with
> automated drift detection and MLOps capabilities.**

![PredictIQ Dashboard](figures/1)End-to-End%20System%20Architecture%20Blueprint-Illustrates%20the%20integration%20between%20the%20React%20client,%20FastAPI%20application%20server,%20ML%20processing%20pipeline,%20and%20SQLite%20database%20storage.png)

---

## 🎯 Overview

PredictIQ is an end-to-end machine learning platform that combines:

- **Advanced ML Pipeline**: 5-phase data science workflow (Preprocessing → RFM
  Engineering → Segmentation → Training → Drift Detection)
- **Real-time Predictions**: WebSocket streaming with live accuracy monitoring
- **AI-Powered Assistant**: Context-aware chatbot using GROQ API (Llama 3.1)
- **MLOps Automation**: One-click model retraining with hot-reload capabilities
- **Production Monitoring**: Prometheus + Grafana dashboards for system
  observability

### 🏆 Key Highlights

- **9-Dimensional Feature Engineering**: Beyond RFM (Recency, Frequency,
  Monetary)
- **5 ML Classifiers**: Logistic Regression, Random Forest, XGBoost, LightGBM,
  Stacking Ensemble
- **Class Imbalance Handling**: SMOTE resampling + proper evaluation (MCC,
  ROC-AUC, PR-AUC)
- **Drift Detection**: Population Stability Index (PSI) with automated
  retraining alerts
- **Explainability**: SHAP values for model interpretability
- **Real-time Streaming**: 5 predictions/second via WebSocket
- **Docker Deployment**: One-command production setup with PostgreSQL, Redis,
  Nginx

---

## 📊 Model Performance

| Model                   | Accuracy | ROC-AUC   | MCC       | CV ROC-AUC        | Status       |
| ----------------------- | -------- | --------- | --------- | ----------------- | ------------ |
| **Random Forest**       | 66.4%    | 0.714     | 0.334     | **0.763** ± 0.027 | ⭐ Best CV   |
| **Logistic Regression** | 67.0%    | **0.735** | **0.360** | 0.738 ± 0.004     | ⭐ Best Test |
| XGBoost                 | 65.0%    | 0.711     | 0.332     | 0.752 ± 0.027     | ✓ Stable     |
| LightGBM                | 65.0%    | 0.705     | 0.299     | 0.753 ± 0.032     | ✓ Fast       |
| Stacking Ensemble       | 65.2%    | 0.721     | 0.306     | 0.744 ± 0.006     | ✓ Robust     |
| **Baseline (Majority)** | 58.2%    | 0.500     | 0.000     | 0.500 ± 0.000     | ⚠️ No Signal |

> **The F1-Score Paradox**: Baseline achieves 73.6% F1 but 0.0 MCC and 0.5 AUC,
> demonstrating why proper metrics matter for imbalanced data.

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- OR Python 3.11+ & Node.js 20+ (manual setup)

### Option 1: Docker Deployment (Production-Ready)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/predictiq.git
cd predictiq

# 2. Configure environment
cp .env.example .env
nano .env  # Add your GROQ_API_KEY and secure passwords

# 3. Launch entire stack (backend, frontend, database, monitoring)
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

### Option 2: Manual Development Setup

#### Backend Setup

```bash
# Create virtual environment
python -m venv backend/venv
source backend/venv/bin/activate  # On Windows: backend\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run full ML pipeline (data cleaning, feature engineering, clustering, training, drift checks)
python run_pipeline.py

# Start FastAPI server (run from project root)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

---

## 📁 Project Structure

```
predictiq/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── db/
│   │   └── database.py            # SQLAlchemy models
│   ├── pipeline/
│   │   ├── preprocessing.py       # Data cleaning & outlier removal
│   │   ├── rfm_features.py        # Feature engineering (9D)
│   │   ├── segmentation.py        # K-Means, DBSCAN, Hierarchical
│   │   ├── model_training.py      # 5 classifiers + SHAP
│   │   └── drift_detection.py     # PSI calculation
│   ├── tests/                     # Pytest suite (90%+ coverage)
│   ├── Dockerfile                 # Multi-stage production build
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── pages/                 # 7 React pages
│   │   │   ├── Home.jsx           # Dashboard overview
│   │   │   ├── Segments.jsx       # Customer segments
│   │   │   ├── Models.jsx         # Model comparison
│   │   │   ├── Predict.jsx        # Single/batch predictions
│   │   │   ├── Live.jsx           # WebSocket streaming
│   │   │   ├── Drift.jsx          # Drift monitoring
│   │   │   └── Chat.jsx           # AI assistant
│   │   └── components/            # Reusable UI components
│   ├── Dockerfile                 # Nginx + React build
│   └── package.json
├── data/
│   ├── raw/                       # UCI Online Retail dataset
│   ├── processed/                 # Cleaned & feature-engineered CSVs
│   └── plots/                     # 20+ visualization charts
├── models/                        # Serialized models (.pkl)
├── monitoring/
│   ├── prometheus.yml             # Metrics scraping config
│   └── grafana/                   # Dashboard provisioning
├── nginx/
│   └── nginx.conf                 # Reverse proxy + SSL + rate limiting
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # Automated testing & Docker build
├── docker-compose.yml             # Full-stack orchestration
├── .env.example                   # Environment variables template
└── README.md
```

---

## 🔬 ML Pipeline Architecture

### Phase 1: Data Preprocessing

- Remove null CustomerIDs, cancelled transactions
- Filter invalid quantity/prices
- UK cohort isolation (optional)
- IQR outlier detection on Quantity & TotalPrice
- **Result**: 318k rows from 541k raw transactions (58.7% retention)

### Phase 2: RFM Feature Engineering

- **Core RFM**: Recency, Frequency, Monetary
- **Advanced Features**:
  - `AvgOrderValue`: Monetary / Frequency
  - `UniqueProducts`: Distinct StockCodes purchased
  - `ReturnRate`: Cancelled invoices / Total invoices
  - `CustomerLifetimeDays`: First to last purchase span
  - `PurchaseFrequencyMonthly`: Normalized purchase rate
  - `AvgQuantityPerOrder`: Basket size indicator
- Log transformation + StandardScaler normalization

### Phase 3: Customer Segmentation

- **K-Means (K=4)**: Champions, Loyal Customers, At-Risk, Lost Customers
- **DBSCAN**: Density-based outlier detection
- **Hierarchical**: Agglomerative clustering with dendrogram
- **Evaluation**: Elbow method, Silhouette scores, PCA visualization

### Phase 4: Supervised Model Training

- **Timeline Split**: 9 months features → 3 months target
- **Class Imbalance**: SMOTE on training set only
- **Hyperparameter Tuning**: RandomizedSearchCV (3-fold CV)
- **Ensemble**: Stacking with Logistic Regression meta-learner
- **Evaluation**: Confusion matrices, ROC curves, PR curves, SHAP

### Phase 5: Drift Detection

- **PSI Monitoring**: Population Stability Index per feature
- **Thresholds**: <0.1 Stable | 0.1-0.25 Monitor | >0.25 Retrain
- **Automation**: Scheduled checks with alert triggers

---

## 🛠️ API Endpoints

### Core Endpoints

| Method | Endpoint                     | Description                     |
| ------ | ---------------------------- | ------------------------------- |
| GET    | `/api/health`                | System health check             |
| GET    | `/api/dataset/stats`         | Dataset summary statistics      |
| GET    | `/api/segments/overview`     | Segment distribution & trends   |
| GET    | `/api/segments/customers`    | All segmented customers         |
| GET    | `/api/models/metrics`        | Model performance comparison    |
| POST   | `/api/predict/single`        | Single customer prediction      |
| POST   | `/api/predict/batch`         | CSV batch predictions           |
| GET    | `/api/predict/history`       | Last 50 prediction logs         |
| GET    | `/api/drift/status`          | Current drift metrics           |
| POST   | `/api/drift/check`           | Trigger drift calculation       |
| POST   | `/api/models/retrain`        | Background retraining pipeline  |
| GET    | `/api/models/retrain/status` | Retraining progress             |
| GET    | `/api/charts/{name}`         | Visualization charts (PNG)      |
| POST   | `/api/chat`                  | AI assistant (GROQ API)         |
| WS     | `/ws/realtime-predict`       | Real-time streaming predictions |

### Interactive API Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🧪 Testing

### Run Test Suite

```bash
cd backend

# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with markers
pytest -m "not slow" -v
```

### Test Coverage

- **API Endpoints**: 95%+ coverage
- **ML Pipeline**: 90%+ coverage
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Response time validation

---

## 📈 Monitoring & Observability

### Prometheus Metrics

- Request rate, latency, error rate
- Model inference time
- Database connection pool
- WebSocket active connections

### Grafana Dashboards

- System health overview
- API performance metrics
- Model prediction distribution
- Drift alert history

**Access**: `http://localhost:3001` (admin/admin)

---

## 🔐 Security Features

- ✅ Non-root Docker containers
- ✅ Environment variable secrets
- ✅ CORS configuration
- ✅ Rate limiting (Nginx + SlowAPI)
- ✅ SSL/TLS support (production)
- ✅ Security headers (HSTS, CSP, X-Frame-Options)
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ Dependency scanning (Trivy)

---

## 🚢 Deployment

### Cloud Platforms

#### AWS (ECS/Fargate)

```bash
# Build and push images
docker-compose build
docker tag predictiq-backend:latest your-ecr-repo/backend:latest
docker push your-ecr-repo/backend:latest

# Deploy via ECS CLI or Terraform
```

#### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/predictiq-backend
gcloud run deploy predictiq --image gcr.io/PROJECT_ID/predictiq-backend --platform managed
```

#### Azure Container Instances

```bash
az container create --resource-group myResourceGroup \
  --name predictiq --image youracr.azurecr.io/predictiq:latest \
  --dns-name-label predictiq --ports 80 443
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 (Python) and ESLint (JavaScript)
- Write tests for new features
- Update documentation
- Ensure CI/CD pipeline passes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

---

## 🙏 Acknowledgments

- **Dataset**:
  [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Frameworks**: FastAPI, React, scikit-learn, XGBoost, LightGBM
- **Icons**: Lucide React
- **Charts**: Recharts
- **AI**: GROQ API (Llama 3.1)

---

## 📞 Support

- **Documentation**: [docs.predictiq.com](https://docs.predictiq.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/predictiq/issues)
- **Email**: support@predictiq.com

---

## 🗺️ Roadmap

- [ ] Model A/B testing framework
- [ ] Multi-tenancy support
- [ ] Advanced feature store (Feast)
- [ ] Real-time feature computation
- [ ] Model versioning (MLflow)
- [ ] Kubernetes Helm charts
- [ ] GraphQL API
- [ ] Mobile app (React Native)

---

**Made with ❤️ by the PredictIQ Team**

⭐ **Star this repo if you find it useful!** ⭐
