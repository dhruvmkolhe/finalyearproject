import sys
import os

# Insert parent directory of 'backend' to sys.path to allow running as main:app inside target folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import sqlite3
import joblib
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
import shap
import io
import logging
from backend.db.database import init_db as db_init, SessionLocal, PredictionLog, DriftMetric

# Load environment variables from .env if present
dotenv_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(dotenv_path):
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception as e:
        print(f"Error loading .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="PredictIQ REST & WebSocket API",
    description="Backend API for real-time customer segmentation and purchase behavior prediction.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database file path
DB_PATH = os.path.join(os.getcwd(), "backend", "database.db")

# Global variables for models, scalers, and cached data
models = {}
scaler = None
xgb_shap_explainer = None
db_centroids = {}  # Store segment centroids for K-Means mapping
cached_dataset_stats = {}
cached_segment_overview = {}
model_metrics = {}

# Schema definitions
class CustomerFeatures(BaseModel):
    Recency: float = Field(..., description="Days since last purchase")
    Frequency: float = Field(..., description="Total number of unique invoices")
    Monetary: float = Field(..., description="Total monetary spend value")
    AvgOrderValue: float = Field(..., description="Monetary spend divided by Frequency")
    UniqueProducts: float = Field(..., description="Number of unique products purchased")
    ReturnRate: float = Field(..., description="Proportion of cancelled transactions")
    CustomerLifetimeDays: float = Field(..., description="Span of days between first and last purchase")
    PurchaseFrequencyMonthly: float = Field(..., description="Number of purchases scaled to a monthly basis")
    AvgQuantityPerOrder: float = Field(..., description="Average number of units purchased per order")

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

def init_db():
    """
    Initializes SQLite tables for logging predictions and drift using SQLAlchemy.
    """
    db_init()

def load_models_and_scaler():
    """
    Loads all pre-trained classifiers, scaler, and calculates K-Means centroids.
    """
    global scaler, models, xgb_shap_explainer, db_centroids, model_metrics
    models_dir = os.path.join(os.getcwd(), "models")
    
    try:
        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
        # Load 5 classification models
        model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'stacking_ensemble']
        for name in model_names:
            model_path = os.path.join(models_dir, f"{name}.pkl")
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                
        # Initialize SHAP explainer on XGBoost
        if 'xgboost' in models:
            xgb_shap_explainer = shap.TreeExplainer(models['xgboost'])
            
        # Load model metrics JSON
        metrics_path = os.path.join(models_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                model_metrics = json.load(f)
                
    except Exception as e:
        print(f"Error loading models or scaler: {e}")

    # Compute segment centroids from segmented customers dataset
    try:
        seg_customers_path = os.path.join(os.getcwd(), "data", "processed", "segmented_customers.csv")
        if os.path.exists(seg_customers_path):
            df_seg = pd.read_csv(seg_customers_path)
            scaled_cols = [
                'Recency_log_scaled', 'Frequency_log_scaled', 'Monetary_log_scaled', 
                'AvgOrderValue_scaled', 'UniqueProducts_scaled', 'ReturnRate_scaled',
                'CustomerLifetimeDays_scaled', 'PurchaseFrequencyMonthly_scaled', 'AvgQuantityPerOrder_scaled'
            ]
            # Group by segment and take mean of scaled features
            centroids = df_seg.groupby('Segment')[scaled_cols].mean()
            db_centroids = {segment: row.values for segment, row in centroids.iterrows()}
    except Exception as e:
        print(f"Error computing segment centroids: {e}")

def precompute_cached_data():
    """
    Precomputes and caches dataset statistics and segment monthly trends
    to speed up API response times dramatically.
    """
    global cached_dataset_stats, cached_segment_overview
    
    cleaned_csv = os.path.join(os.getcwd(), "data", "processed", "cleaned_retail.csv")
    seg_csv = os.path.join(os.getcwd(), "data", "processed", "segmented_customers.csv")
    
    if os.path.exists(cleaned_csv):
        df_cleaned = pd.read_csv(cleaned_csv)
        df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
        
        # Top 10 products
        top_10 = df_cleaned.groupby('Description').agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum'
        }).sort_values(by='Quantity', ascending=False).head(10).reset_index()
        
        cached_dataset_stats = {
            "total_records": len(df_cleaned),
            "total_customers": int(df_cleaned['CustomerID'].nunique()),
            "total_transactions": int(df_cleaned['InvoiceNo'].nunique()),
            "date_range": {
                "start": df_cleaned['InvoiceDate'].min().strftime("%Y-%m-%d"),
                "end": df_cleaned['InvoiceDate'].max().strftime("%Y-%m-%d")
            },
            "country_breakdown": df_cleaned['Country'].value_counts().to_dict(),
            "top_products": [
                {
                    "description": row['Description'],
                    "quantity": int(row['Quantity']),
                    "revenue": float(row['TotalPrice'])
                }
                for _, row in top_10.iterrows()
            ]
        }
        
    if os.path.exists(seg_csv) and os.path.exists(cleaned_csv):
        df_seg = pd.read_csv(seg_csv)
        df_cleaned = pd.read_csv(cleaned_csv)
        df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
        df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(str)
        df_seg['CustomerID'] = df_seg['CustomerID'].astype(str)
        
        # Segment distribution
        seg_counts = df_seg['Segment'].value_counts()
        seg_pct = df_seg['Segment'].value_counts(normalize=True) * 100
        
        # Centroids
        raw_centroids = df_seg.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'AvgOrderValue': 'mean',
            'UniqueProducts': 'mean',
            'ReturnRate': 'mean'
        }).reset_index().to_dict(orient='records')
        
        # Monthly trend per segment
        # Merge segment classification to transactions
        df_merged = df_cleaned.merge(df_seg[['CustomerID', 'Segment']], on='CustomerID', how='inner')
        df_merged['Month'] = df_merged['InvoiceDate'].dt.to_period('M').astype(str)
        
        monthly_grouped = df_merged.groupby(['Month', 'Segment']).agg({
            'TotalPrice': 'sum',
            'CustomerID': 'nunique'
        }).reset_index()
        
        # Structure trend data for Recharts
        trend_pivot_spend = monthly_grouped.pivot(index='Month', columns='Segment', values='TotalPrice').fillna(0).to_dict(orient='index')
        trend_pivot_cust = monthly_grouped.pivot(index='Month', columns='Segment', values='CustomerID').fillna(0).to_dict(orient='index')
        
        trend_data = []
        for month in sorted(trend_pivot_spend.keys()):
            trend_data.append({
                "month": month,
                "spend": trend_pivot_spend[month],
                "customers": trend_pivot_cust[month]
            })
            
        cached_segment_overview = {
            "distribution": [
                {
                    "segment": seg,
                    "count": int(seg_counts[seg]),
                    "percentage": float(seg_pct[seg])
                }
                for seg in seg_counts.index
            ],
            "centroids": raw_centroids,
            "monthly_trend": trend_data
        }

@app.on_event("startup")
async def startup_event():
    init_db()
    load_models_and_scaler()
    precompute_cached_data()
    # Pre-cache Mermaid diagrams as PNGs
    for name in DIAGRAMS_CODE.keys():
        try:
            get_mermaid_png_path(name)
        except Exception as e:
            logger.error(f"Startup pre-cache failed for {name}: {e}")

# ----------------- REST ENDPOINTS -----------------

@app.get("/api/health")
def get_health():
    """
    Returns API health check and models load status.
    """
    models_status = {
        name: (name in models) for name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'stacking_ensemble']
    }
    models_status['scaler'] = (scaler is not None)
    
    is_healthy = all(models_status.values())
    
    return {
        "success": True,
        "data": {
            "status": "healthy" if is_healthy else "degraded",
            "models_loaded": models_status
        },
        "error": None
    }

@app.get("/api/dataset/stats")
def get_dataset_stats():
    """
    Returns precomputed summaries of the Online Retail dataset.
    """
    if not cached_dataset_stats:
        return {"success": False, "data": None, "error": "Dataset stats not precomputed. Ensure cleaned_retail.csv exists."}
    return {
        "success": True,
        "data": cached_dataset_stats,
        "error": None
    }

@app.get("/api/segments/overview")
def get_segments_overview():
    """
    Returns precomputed segment distributions, centroid details, and month-over-month trends.
    """
    if not cached_segment_overview:
        return {"success": False, "data": None, "error": "Segment overview not precomputed. Run segmentation first."}
    return {
        "success": True,
        "data": cached_segment_overview,
        "error": None
    }

@app.get("/api/segments/customers")
def get_segments_customers():
    """
    Returns the list of all segmented customer profiles.
    """
    seg_csv = os.path.join(os.getcwd(), "data", "processed", "segmented_customers.csv")
    if not os.path.exists(seg_csv):
        return {"success": False, "data": [], "error": "Segmented customer file not found."}
    try:
        df = pd.read_csv(seg_csv)
        df['CustomerID'] = df['CustomerID'].astype(str)
        return {
            "success": True,
            "data": df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'ReturnRate', 'Segment']].to_dict(orient='records'),
            "error": None
        }
    except Exception as e:
        return {"success": False, "data": [], "error": str(e)}

@app.get("/api/models/metrics")
def get_models_metrics():
    """
    Returns validation metrics of trained classifiers, recommending the best performer.
    """
    if not model_metrics:
        return {"success": False, "data": None, "error": "Model metrics not found. Run model training first."}
    
    # Recommend model with highest cross-validated ROC-AUC
    best_model = None
    best_auc = -1.0
    for m_name, m_val in model_metrics.items():
        if m_val.get('cv_roc_auc_mean', 0) > best_auc:
            best_auc = m_val['cv_roc_auc_mean']
            best_model = m_name
            
    return {
        "success": True,
        "data": {
            "metrics": model_metrics,
            "recommended_model": best_model,
            "recommendation_reason": f"Model recommends '{best_model.replace('_', ' ').title()}' due to highest 5-Fold Cross-Validation ROC-AUC of {best_auc:.4f}."
        },
        "error": None
    }

@app.post("/api/predict/single")
def predict_single(features: CustomerFeatures):
    """
    Accepts RFM values for a single customer, returns predictions for 5 models,
    assigns a segment using nearest centroid, computes top 3 SHAP importances, and logs to SQLite.
    """
    if scaler is None or not models:
        raise HTTPException(status_code=503, detail="Models or scalers are not loaded on server.")
        
    try:
        # Preprocessing & Transformations
        recency_log = np.log1p(features.Recency)
        frequency_log = np.log1p(features.Frequency)
        monetary_log = np.log1p(features.Monetary)
        
        feature_names = [
            'Recency_log', 'Frequency_log', 'Monetary_log', 
            'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
            'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
        ]
        raw_values = [
            recency_log, frequency_log, monetary_log, 
            features.AvgOrderValue, features.UniqueProducts, features.ReturnRate,
            features.CustomerLifetimeDays, features.PurchaseFrequencyMonthly, features.AvgQuantityPerOrder
        ]
        
        # Scale
        scaled_values = scaler.transform([raw_values])[0]
        scaled_df = pd.DataFrame([scaled_values], columns=feature_names)
        
        # Segment Assignment based on Nearest Scaled Centroid
        predicted_segment = "Unknown"
        min_dist = float('inf')
        if db_centroids:
            for segment, centroid in db_centroids.items():
                dist = np.linalg.norm(scaled_values - centroid)
                if dist < min_dist:
                    min_dist = dist
                    predicted_segment = segment
                    
        # Predictions & Probabilities
        preds = {}
        probs = {}
        for name, model in models.items():
            preds[name] = int(model.predict(scaled_df)[0])
            probs[name] = float(model.predict_proba(scaled_df)[0][1])
            
        # SHAP calculation (on XGBoost)
        top_shap_features = []
        if xgb_shap_explainer is not None:
            shap_output = xgb_shap_explainer(scaled_df)
            # Handle shape dimensions for binary classification (3D vs 2D)
            if len(shap_output.shape) == 3:
                shap_vals = shap_output.values[0, :, 1]
            else:
                shap_vals = shap_output.values[0]
                
            # Create feature influence list
            shap_list = []
            for name, val in zip(feature_names, shap_vals):
                shap_list.append({
                    "feature": name,
                    "shap_value": float(val),
                    "influence": "positive" if val > 0 else "negative"
                })
            # Sort by absolute SHAP value descending
            top_shap_features = sorted(shap_list, key=lambda x: abs(x['shap_value']), reverse=True)[:3]
            
        # Log prediction to SQLite database using SQLAlchemy
        db = SessionLocal()
        try:
            log_entry = PredictionLog(
                timestamp=datetime.now().isoformat(),
                recency=features.Recency,
                frequency=features.Frequency,
                monetary=features.Monetary,
                avg_order_value=features.AvgOrderValue,
                unique_products=features.UniqueProducts,
                return_rate=features.ReturnRate,
                customer_lifetime_days=features.CustomerLifetimeDays,
                purchase_frequency_monthly=features.PurchaseFrequencyMonthly,
                avg_quantity_per_order=features.AvgQuantityPerOrder,
                predicted_segment=predicted_segment,
                lr_pred=preds['logistic_regression'],
                lr_prob=probs['logistic_regression'],
                rf_pred=preds['random_forest'],
                rf_prob=probs['random_forest'],
                xgb_pred=preds['xgboost'],
                xgb_prob=probs['xgboost'],
                lgb_pred=preds['lightgbm'],
                lgb_prob=probs['lightgbm'],
                stacking_pred=preds['stacking_ensemble'],
                stacking_prob=probs['stacking_ensemble']
            )
            db.add(log_entry)
            db.commit()
        except Exception as db_err:
            logger.error(f"Database logging failed: {db_err}")
            db.rollback()
        finally:
            db.close()
        
        return {
            "success": True,
            "data": {
                "predictions": {name: {"label": preds[name], "probability": probs[name]} for name in preds},
                "assigned_segment": predicted_segment,
                "top_shap_features": top_shap_features
            },
            "error": None
        }
        
    except Exception as e:
        return {"success": False, "data": None, "error": f"Inference failed: {str(e)}"}

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts customer CSV file, runs batch predictions using the Stacking Ensemble,
    assigns segments, and returns the modified CSV.
    """
    if scaler is None or 'stacking_ensemble' not in models:
        raise HTTPException(status_code=503, detail="Stacking Ensemble or scale file not loaded on server.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verify required columns are present
        required_cols = [
            'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
            'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV file missing required feature columns: {missing}")
            
        # Feature transformation
        df_proc = df.copy()
        df_proc['Recency_log'] = np.log1p(df_proc['Recency'])
        df_proc['Frequency_log'] = np.log1p(df_proc['Frequency'])
        df_proc['Monetary_log'] = np.log1p(df_proc['Monetary'])
        
        feature_names = [
            'Recency_log', 'Frequency_log', 'Monetary_log', 
            'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
            'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
        ]
        
        # Scale
        X_scaled = scaler.transform(df_proc[feature_names])
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Segment Centroids assignment
        assigned_segments = []
        if db_centroids:
            for row in X_scaled:
                min_dist = float('inf')
                best_seg = "Unknown"
                for segment, centroid in db_centroids.items():
                    dist = np.linalg.norm(row - centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_seg = segment
                assigned_segments.append(best_seg)
        else:
            assigned_segments = ["Unknown"] * len(df)
            
        # Inference using Stacking Ensemble
        stack_model = models['stacking_ensemble']
        preds = stack_model.predict(X_scaled_df)
        probs = stack_model.predict_proba(X_scaled_df)[:, 1]
        
        # Append to output dataframe
        df['Predicted_Segment'] = assigned_segments
        df['Predicted_Purchase_Label'] = preds
        df['Purchase_Probability'] = probs
        
        # Export back to CSV string
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        return StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"}
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction processing failed: {str(e)}")

@app.get("/api/predict/history")
def get_prediction_history():
    """
    Retrieves the last 50 customer prediction records from SQLite using SQLAlchemy.
    """
    try:
        db = SessionLocal()
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(50).all()
        db.close()
        
        history = []
        for log in logs:
            history.append({
                "id": log.id,
                "timestamp": log.timestamp,
                "recency": log.recency,
                "frequency": log.frequency,
                "monetary": log.monetary,
                "avg_order_value": log.avg_order_value,
                "unique_products": log.unique_products,
                "return_rate": log.return_rate,
                "customer_lifetime_days": log.customer_lifetime_days,
                "purchase_frequency_monthly": log.purchase_frequency_monthly,
                "avg_quantity_per_order": log.avg_quantity_per_order,
                "predicted_segment": log.predicted_segment,
                "lr_pred": log.lr_pred,
                "lr_prob": log.lr_prob,
                "rf_pred": log.rf_pred,
                "rf_prob": log.rf_prob,
                "xgb_pred": log.xgb_pred,
                "xgb_prob": log.xgb_prob,
                "lgb_pred": log.lgb_pred,
                "lgb_prob": log.lgb_prob,
                "stacking_pred": log.stacking_pred,
                "stacking_prob": log.stacking_prob
            })
            
        return {
            "success": True,
            "data": history,
            "error": None
        }
    except Exception as e:
        return {"success": False, "data": None, "error": f"Failed to retrieve history: {str(e)}"}

@app.get("/api/drift/status")
def get_drift_status():
    """
    Reads the latest computed PSI drift results from SQLite and fallback JSON drift report using SQLAlchemy.
    """
    report_json_path = os.path.join(os.getcwd(), "data", "processed", "drift_report.json")
    db_metrics = []
    
    try:
        db = SessionLocal()
        latest_metric = db.query(DriftMetric).order_by(DriftMetric.id.desc()).first()
        if latest_metric:
            latest_ts = latest_metric.timestamp
            metrics_rows = db.query(DriftMetric).filter(DriftMetric.timestamp == latest_ts).all()
            for r in metrics_rows:
                db_metrics.append({
                    "feature": r.feature_name,
                    "psi_value": r.psi_value,
                    "status": r.status,
                    "training_mean": r.training_mean,
                    "production_mean": r.production_mean
                })
        db.close()
    except Exception as e:
        print(f"Error querying database drift: {e}")
        
    # Fallback to JSON file if database query failed or is empty
    if not db_metrics and os.path.exists(report_json_path):
        try:
            with open(report_json_path, 'r') as f:
                report = json.load(f)
                return {
                    "success": True,
                    "data": report,
                    "error": None
                }
        except Exception as e:
            return {"success": False, "data": None, "error": f"Failed reading drift JSON: {str(e)}"}
            
    if db_metrics:
        # Determine overall status
        overall = "Stable"
        for m in db_metrics:
            if m['status'] == "Retrain Alert":
                overall = "Retrain Alert"
            elif m['status'] == "Monitor" and overall == "Stable":
                overall = "Monitor"
                
        return {
            "success": True,
            "data": {
                "overall_status": overall,
                "timestamp": datetime.now().isoformat(),
                "metrics": db_metrics
            },
            "error": None
        }
        
    return {"success": False, "data": None, "error": "No drift data available. Run drift detection script."}

@app.post("/api/drift/check")
def trigger_drift_check():
    """
    Triggers the Population Stability Index (PSI) data drift check pipeline.
    """
    from backend.pipeline.drift_detection import run_drift_detection
    try:
        report = run_drift_detection()
        return {
            "success": True,
            "data": report,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Failed running drift calculation: {str(e)}"
        }

# MLOps Retraining Process State Tracking
retraining_status = {
    "status": "idle",       # "idle", "running", "success", "failed"
    "progress": 0,          # 0 to 100
    "error": None,
    "started_at": None,
    "finished_at": None
}

def execute_pipeline_retraining():
    global retraining_status
    retraining_status["status"] = "running"
    retraining_status["progress"] = 5
    retraining_status["error"] = None
    retraining_status["started_at"] = datetime.now().isoformat()
    retraining_status["finished_at"] = None

    try:
        # Phase 1: Preprocessing
        logger.info("MLOps: Retraining Phase 1 - Running Preprocessing...")
        retraining_status["progress"] = 15
        from backend.pipeline.preprocessing import run_preprocessing
        run_preprocessing(
            raw_path="data/raw/OnlineRetail.xlsx",
            output_path="data/processed/cleaned_retail.csv",
            filter_uk_only=True
        )

        # Phase 2: Feature Engineering
        logger.info("MLOps: Retraining Phase 2 - Running RFM Feature Engineering...")
        retraining_status["progress"] = 35
        from backend.pipeline.rfm_features import run_rfm_engineering
        run_rfm_engineering(
            cleaned_path="data/processed/cleaned_retail.csv",
            raw_path="data/raw/OnlineRetail.xlsx",
            output_path="data/processed/rfm_features.csv",
            scaler_path="models/scaler.pkl",
            correlation_plot_path="data/plots/rfm_correlation.png"
        )

        # Phase 3: Customer Segmentation
        logger.info("MLOps: Retraining Phase 3 - Running Unsupervised Segmentation...")
        retraining_status["progress"] = 55
        from backend.pipeline.segmentation import run_segmentation
        run_segmentation(
            features_path="data/processed/rfm_features.csv",
            output_path="data/processed/segmented_customers.csv",
            plots_dir="data/plots"
        )

        # Phase 4: Model Classifier Training
        logger.info("MLOps: Retraining Phase 4 - Running Supervised Classifier Training...")
        retraining_status["progress"] = 75
        from backend.pipeline.model_training import run_model_training
        run_model_training(
            cleaned_path="data/processed/cleaned_retail.csv",
            raw_path="data/raw/OnlineRetail.xlsx",
            models_dir="models",
            plots_dir="data/plots"
        )

        # Phase 5: Data Drift Check
        logger.info("MLOps: Retraining Phase 5 - Recalculating Data Drift Baseline...")
        retraining_status["progress"] = 90
        from backend.pipeline.drift_detection import run_drift_detection
        run_drift_detection(
            reference_path="data/processed/rfm_features.csv",
            db_path="backend/database.db",
            simulate_drift=True
        )

        # Reload newly created model packages and scaler dynamically in RAM
        logger.info("MLOps: Dynamically reloading trained model packages into server RAM...")
        load_models_and_scaler()

        # Refresh cached dashboard statistics and charts
        logger.info("MLOps: Refreshing dashboard cached statistics...")
        precompute_cached_data()

        retraining_status["status"] = "success"
        retraining_status["progress"] = 100
        retraining_status["finished_at"] = datetime.now().isoformat()
        logger.info("MLOps: Dynamic retraining pipeline executed and reloaded successfully.")

    except Exception as e:
        logger.error(f"MLOps Retraining Pipeline failed: {str(e)}", exc_info=True)
        retraining_status["status"] = "failed"
        retraining_status["error"] = str(e)
        retraining_status["finished_at"] = datetime.now().isoformat()

@app.post("/api/models/retrain")
def trigger_pipeline_retrain(background_tasks: BackgroundTasks):
    """
    Triggers the full 5-phase data science pipeline in the background and reloads models.
    """
    global retraining_status
    if retraining_status["status"] == "running":
        return {
            "success": False,
            "error": "A retraining pipeline execution is already in progress.",
            "data": retraining_status
        }
    
    # Reset status and run as FastAPI background task
    background_tasks.add_task(execute_pipeline_retraining)
    return {
        "success": True,
        "message": "Model retraining pipeline launched in background.",
        "data": {
            "status": "running",
            "progress": 5,
            "error": None
        }
    }

@app.get("/api/models/retrain/status")
def get_pipeline_retrain_status():
    """
    Returns the current status of the background model retraining pipeline.
    """
    return {
        "success": True,
        "data": retraining_status,
        "error": None
    }

@app.get("/api/charts/{chart_name}")
def get_chart(chart_name: str):
    """
    Serves generated PNG visualization charts from the data/plots directory.
    """
    chart_mapping = {
        "roc": "roc_curve_comparison.png",
        "pr": "pr_curve_comparison.png",
        "shap_summary": "shap_summary.png",
        "shap_bar": "shap_bar.png",
        "pca": "customer_segments_pca.png",
        "rf_importance": "rf_feature_importances.png",
        "kmeans_eval": "kmeans_evaluation.png",
        "dendrogram": "hierarchical_dendrogram.png",
        "dbscan_kdist": "dbscan_k_distance.png",
        
        # Confusion matrices
        "confusion_lr": "confusion_matrix_logistic_regression.png",
        "confusion_rf": "confusion_matrix_random_forest.png",
        "confusion_xgb": "confusion_matrix_xgboost.png",
        "confusion_lgb": "confusion_matrix_lightgbm.png",
        "confusion_stacking": "confusion_matrix_stacking_ensemble.png"
    }
    
    filename = chart_mapping.get(chart_name)
    if not filename:
        raise HTTPException(status_code=404, detail="Invalid chart name. Refer to API documents.")
        
    chart_path = os.path.join(os.getcwd(), "data", "plots", filename)
    
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail=f"Chart image file not found. Ensure models have run.")
        
    return FileResponse(chart_path, media_type="image/png")

    return FileResponse(chart_path, media_type="image/png")

# --- DIAGRAMS PRE-CACHING AND SERVING ---
DIAGRAMS_CODE = {
    "sys_arch": """graph TD
    subgraph Client ["React Frontend Client"]
        A[SaaS Analytics Dashboard]
        A1[Recharts Dashboard]
        A2[Live Prediction Streamer]
        A3[Model Arena Comparator]
        A4[Data Drift Monitor]
    end

    subgraph Server ["FastAPI Application Server"]
        B[FastAPI REST Router]
        C[WebSocket Endpoint]
        D[MLOps Retraining Coordinator]
    end

    subgraph DataPipeline ["ML Pipeline Layer"]
        E[Preprocessing & Outlier Filter]
        F[RFM & Feature Engineering]
        G[Unsupervised Segmentation]
        H[Supervised Model Training]
        I[Data Drift Engine]
    end

    subgraph Storage ["Database & Storage"]
        J[(SQLite Database)]
        K[joblib Model Packages]
        L[Processed CSV Datasets]
        M[Visualization Plot Files]
    end

    A --> B
    A2 <--> C
    A3 --> D
    
    B --> L
    B --> M
    D --> E
    
    E --> F
    F --> G
    G --> H
    H --> I
    
    H -.-> K
    F -.-> L
    H -.-> M
    I -.-> J
    B -.-> J""",

    "pipeline_flow": """flowchart LR
    raw[OnlineRetail.xlsx] --> P1[Phase 1: Preprocessing & Cleaning]
    P1 -->|cleaned_retail.csv| P2[Phase 2: RFM Feature Engineering]
    P2 -->|rfm_features.csv| P3[Phase 3: Unsupervised Segmentation]
    P3 -->|segmented_customers.csv| P4[Phase 4: Supervised Classifier Arena]
    P4 -->|Models / Metrics| P5[Phase 5: Data Drift Monitor]
    
    subgraph P3_Details [Clustering Algorithms]
        KMeans[K-Means K=4]
        DBSCAN[DBSCAN]
        Hierarchical[Agglomerative]
    end
    P3 -.-> P3_Details
    
    subgraph P4_Details [Supervised Arena]
        SMOTE[SMOTE Resampling]
        Tuning[RandomizedSearchCV]
        Stacking[Stacking Ensemble]
    end
    P4 -.-> P4_Details""",

    "preprocessing_flow": """flowchart TD
    Start([Raw Dataset: ~541k Rows]) --> F1{Null Customer ID?}
    F1 -- Yes --> DropNull[Drop Rows]
    F1 -- No --> F2{Cancelled Invoice?<br/>InvoiceNo starts with 'C'}
    F2 -- Yes --> DropCancelled[Exclude Cancellations]
    F2 -- No --> F3{Quantity <= 0 OR<br/>UnitPrice <= 0?}
    F3 -- Yes --> DropZero[Exclude Bad Price/Qty]
    F3 -- No --> F4{Filter UK Only?}
    F4 -- Yes --> UKFilter[Retain Country == 'United Kingdom']
    F4 -- No --> F5[Derived Feature: TotalPrice = Qty * Price]
    UKFilter --> F5
    
    F5 --> IQR_Outlier{Outliers via IQR?}
    IQR_Outlier -->|Quantity Outliers| DropOut1[Filter Qty Bounds]
    IQR_Outlier -->|TotalPrice Outliers| DropOut2[Filter Price Bounds]
    
    DropOut1 --> Finish([Cleaned Retail CSV: ~318k Rows])
    DropOut2 --> Finish""",

    "training_protocol": """flowchart TD
    raw_data[Cleaned Retail Data] --> timeline{Split Timeline}
    timeline -->|First 9 Months| FeatPeriod[Feature Engineering: X]
    timeline -->|Last 3 Months| TargetPeriod[Target Generation: y]
    
    FeatPeriod & TargetPeriod --> Merge[Supervised Dataset]
    
    Merge --> TrainTest{Stratified 80/20 Split}
    
    TrainTest -->|20% Holdout Test Set| TestSet[Scaled via Training Scaler]
    TrainTest -->|80% Training Set| TrainSet[Fit StandardScaler]
    
    TrainSet --> SMOTE[Apply SMOTE Resampling<br/>To Training Only]
    SMOTE --> Resampled[Balanced Training Set]
    
    Resampled --> Tuning[RandomizedSearchCV<br/>3-Fold Cross-Validation]
    Tuning --> BestModels[Optimized Classifiers]
    
    BestModels --> FitStack[Train Stacking Ensemble<br/>Meta-Learner: LogReg]
    FitStack --> Evaluation[Evaluate on 20% Holdout Test Set]""",

    "ws_sequence": """sequenceDiagram
    autonumber
    actor Client as React Web Client
    participant API as FastAPI WebSocket Server
    participant DB as SQLite Database
    
    Client->>API: Establish Connection
    Note over API: Load Serialized Scaler & Stacking Model
    
    loop Stream Records (Every 200ms)
        API->>API: Read next row from Test Dataset
        API->>API: Log-transform & Scale features
        API->>API: Find nearest centroid (Segment)
        API->>API: Predict purchase label & probability
        API->>API: Update rolling correctness metrics
        API-->>Client: Stream predictions (JSON payload)
        Client->>Client: Render dynamic charts
    end
    
    Client->>API: Close connection"""
}

def get_mermaid_png_path(name: str) -> str:
    import base64
    import urllib.request
    import urllib.error
    
    cache_dir = os.path.join(os.getcwd(), "data", "plots", "mermaid")
    os.makedirs(cache_dir, exist_ok=True)
    png_path = os.path.join(cache_dir, f"{name}.png")
    
    if os.path.exists(png_path):
        return png_path
        
    code = DIAGRAMS_CODE.get(name)
    if not code:
        raise ValueError(f"Unknown diagram: {name}")
        
    json_obj = {
        "code": code,
        "mermaid": {
            "theme": "neutral"
        }
    }
    json_str = json.dumps(json_obj)
    encoded = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8').rstrip('=')
    url = f"https://mermaid.ink/img/{encoded}"
    
    try:
        logger.info(f"Caching diagram {name} from mermaid.ink...")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            with open(png_path, 'wb') as f:
                f.write(response.read())
        logger.info(f"Successfully cached diagram {name} to {png_path}")
    except Exception as e:
        logger.error(f"Failed to fetch diagram {name} from mermaid.ink: {e}")
        raise e
        
    return png_path

@app.get("/api/diagrams/png/{name}")
def get_diagram_png(name: str):
    try:
        path = get_mermaid_png_path(name)
        return FileResponse(path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load diagram: {str(e)}")

@app.get("/api/diagrams/download-png/{name}")
def download_diagram_png(name: str):
    try:
        path = get_mermaid_png_path(name)
        return FileResponse(
            path, 
            media_type="image/png", 
            headers={"Content-Disposition": f"attachment; filename=Figure_{name}.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download diagram: {str(e)}")

@app.get("/diagrams")
def get_diagrams_page():
    """
    Serves the publication figures console HTML page.
    """
    html_path = os.path.join(os.getcwd(), "PredictIQ_Publication_Figures.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Publication figures console HTML file not found.")

@app.post("/api/chat")
async def chat_with_groq(request: ChatRequest):
    """
    Accepts user message and history, injects PredictIQ system context, and queries Groq API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {
            "success": False,
            "reply": None,
            "error": "GROQ_API_KEY is not configured on the server. Please check your .env file."
        }

    # Construct context summary for LLM — trimmed to avoid oversized payloads
    def trim_dataset_stats(stats: dict) -> dict:
        if not stats:
            return {}
        trimmed = dict(stats)
        # Keep only top 10 countries to avoid massive payloads
        if "country_breakdown" in trimmed:
            cb = trimmed["country_breakdown"]
            top10 = dict(sorted(cb.items(), key=lambda x: x[1], reverse=True)[:10])
            trimmed["country_breakdown"] = top10
        return trimmed

    def trim_segment_overview(overview: dict) -> dict:
        if not overview:
            return {}
        trimmed = dict(overview)
        # Keep only last 3 months of trend data
        if "monthly_trend" in trimmed and isinstance(trimmed["monthly_trend"], list):
            trimmed["monthly_trend"] = trimmed["monthly_trend"][-3:]
        return trimmed

    dataset_summary = json.dumps(trim_dataset_stats(cached_dataset_stats), indent=2) if cached_dataset_stats else "Not precomputed yet."
    segments_summary = json.dumps(trim_segment_overview(cached_segment_overview), indent=2) if cached_segment_overview else "Not precomputed yet."
    metrics_summary = json.dumps(model_metrics, indent=2) if model_metrics else "Not precomputed yet."

    system_message = (
        "You are PredictIQ Business Intelligence Assistant - a specialized AI that ONLY answers questions about THIS company's customer data and business analytics.\n\n"
        "STRICT RULES:\n"
        "1. ONLY answer questions about the customer database, segments, products, and ML models shown below\n"
        "2. REFUSE to answer general knowledge, world events, coding help, or ANY topic outside this business data\n"
        "3. If asked about unrelated topics, politely redirect: 'I can only help with questions about your customer data and business analytics. Please ask about segments, products, models, or customer behavior.'\n"
        "4. Stay focused on retail business insights, marketing strategies, and predictive analytics\n\n"
        "=== YOUR COMPANY'S LIVE DATA ===\n\n"
        f"DATASET STATISTICS:\n{dataset_summary}\n\n"
        f"CUSTOMER SEGMENTS:\n{segments_summary}\n\n"
        f"ML MODEL PERFORMANCE:\n{metrics_summary}\n\n"
        "=== END OF DATA ===\n\n"
        "HOW TO RESPOND:\n"
        "- Use specific numbers from the data above\n"
        "- Provide actionable business insights\n"
        "- Suggest marketing strategies based on segments\n"
        "- Explain model metrics in business terms\n"
        "- Use Markdown formatting (bold, lists, tables)\n"
        "- Keep answers concise (2-3 paragraphs) unless detailed analysis is requested\n\n"
        "EXAMPLE ACCEPTABLE QUESTIONS:\n"
        "✓ 'How many Champions customers do we have?'\n"
        "✓ 'Which products generate the most revenue?'\n"
        "✓ 'What's the best model for prediction?'\n"
        "✓ 'Suggest a campaign for At-Risk customers'\n\n"
        "EXAMPLE UNACCEPTABLE QUESTIONS (REFUSE THESE):\n"
        "✗ 'What's the weather today?'\n"
        "✗ 'Write me Python code'\n"
        "✗ 'Tell me about world history'\n"
        "✗ 'Help me with my homework'\n\n"
        "Remember: You are a BUSINESS ANALYST, not a general assistant!"
    )

    # Format messages for Groq API
    messages = [{"role": "system", "content": system_message}]
    
    # Append conversation history
    for msg in request.history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ["user", "assistant"]:
            messages.append({"role": role, "content": content})
            
    # Append the new user message
    messages.append({"role": "user", "content": request.message})

    # Call Groq API using urllib
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        
        # We use llama-3.1-8b-instant which is standard and fast.
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024
        }
        
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            reply = res_data["choices"][0]["message"]["content"]
            
        return {
            "success": True,
            "reply": reply,
            "error": None
        }
    except Exception as e:
        # Fallback to llama3-8b-8192 if llama-3.1-8b-instant is not available or errors out
        try:
            payload["model"] = "llama3-8b-8192"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                reply = res_data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "reply": reply,
                "error": None
            }
        except Exception as e2:
            logger.error(f"Groq API call failed: {e} | Fallback: {e2}")
            return {
                "success": False,
                "reply": None,
                "error": f"Failed to get response from AI Chatbot: {str(e2)}"
            }

# ----------------- WEBSOCKET ENDPOINT -----------------

@app.websocket("/ws/realtime-predict")
async def websocket_realtime_predict(websocket: WebSocket):
    """
    Accepts WebSocket connections, streams test customer records row-by-row
    with 200ms delay, and streams back predictions, segment classifications, and running accuracy.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for real-time predictions.")
    
    # Load supervised data and extract test split
    sup_csv = os.path.join(os.getcwd(), "data", "processed", "supervised_data.csv")
    if not os.path.exists(sup_csv):
        await websocket.send_json({"error": "Supervised dataset not found on server. Cannot run live simulation."})
        await websocket.close()
        return
        
    try:
        # Load dataset
        df_sup = pd.read_csv(sup_csv)
        df_sup['CustomerID'] = df_sup['CustomerID'].astype(str)
        
        # Recreate test split (80/20 stratified split using same random state 42)
        feature_cols = [
            'Recency_log', 'Frequency_log', 'Monetary_log', 
            'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
            'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
        ]
        _, X_test_raw, _, y_test = train_test_split(
            df_sup, df_sup['Target'].values, test_size=0.2, random_state=42, stratify=df_sup['Target'].values
        )
        
        logger.info(f"Loaded {len(X_test_raw)} records for real-time WebSocket simulation.")
        
        # Verify Stacking Ensemble and Scaler are available
        stack_model = models.get('stacking_ensemble')
        if not stack_model or scaler is None:
            await websocket.send_json({"error": "Stacking ensemble or scaler is not loaded. Cannot run predictions."})
            await websocket.close()
            return
            
        total_processed = 0
        correct_predictions = 0
        
        # Loop through the test records row-by-row
        for idx, row in X_test_raw.iterrows():
            total_processed += 1
            
            customer_id = str(row['CustomerID'])
            true_label = int(row['Target'])
            
            # Extract features
            recency = float(row['Recency'])
            frequency = float(row['Frequency'])
            monetary = float(row['Monetary'])
            avg_order_value = float(row['AvgOrderValue'])
            unique_products = float(row['UniqueProducts'])
            return_rate = float(row['ReturnRate'])
            customer_lifetime_days = float(row['CustomerLifetimeDays'])
            purchase_frequency_monthly = float(row['PurchaseFrequencyMonthly'])
            avg_quantity_per_order = float(row['AvgQuantityPerOrder'])
            
            # Values for prediction (log-transform Recency, Frequency, Monetary dynamically)
            recency_log = np.log1p(recency)
            frequency_log = np.log1p(frequency)
            monetary_log = np.log1p(monetary)
            features_raw = [
                recency_log, frequency_log, monetary_log, 
                avg_order_value, unique_products, return_rate,
                customer_lifetime_days, purchase_frequency_monthly, avg_quantity_per_order
            ]
            
            # Scale
            scaled_feat = scaler.transform([features_raw])[0]
            scaled_df = pd.DataFrame([scaled_feat], columns=feature_cols)
            
            # Nearest centroid segment mapping
            predicted_segment = "Unknown"
            min_dist = float('inf')
            if db_centroids:
                for segment, centroid in db_centroids.items():
                    dist = np.linalg.norm(scaled_feat - centroid)
                    if dist < min_dist:
                        min_dist = dist
                        predicted_segment = segment
                        
            # Inference using Stacking Ensemble
            pred_label = int(stack_model.predict(scaled_df)[0])
            pred_prob = float(stack_model.predict_proba(scaled_df)[0][1])
            
            # Accuracy metric update
            is_correct = (pred_label == true_label)
            if is_correct:
                correct_predictions += 1
            running_accuracy = (correct_predictions / total_processed) * 100
            
            # Build and send JSON payload
            payload = {
                "customer_id": customer_id,
                "features": {
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary,
                    "AvgOrderValue": avg_order_value,
                    "UniqueProducts": unique_products,
                    "ReturnRate": return_rate,
                    "CustomerLifetimeDays": customer_lifetime_days,
                    "PurchaseFrequencyMonthly": purchase_frequency_monthly,
                    "AvgQuantityPerOrder": avg_quantity_per_order
                },
                "true_label": true_label,
                "prediction": pred_label,
                "probability": pred_prob,
                "segment": predicted_segment,
                "running_accuracy": float(running_accuracy),
                "total_processed": total_processed
            }
            
            # Send message
            await websocket.send_json(payload)
            
            # Non-blocking sleep delay
            await asyncio.sleep(0.2) # 200ms delay
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket execution error: {e}")
        try:
            await websocket.send_json({"error": f"Simulation interrupted: {str(e)}"})
        except Exception:
            pass
        finally:
            await websocket.close()
