import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from backend.db.database import SessionLocal, DriftMetric, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between expected (training) and actual (production) distributions.
    """
    # Use percentiles based on training (expected) data to determine bin edges
    percentiles = np.linspace(0, 100, num_bins + 1)
    thresholds = np.percentile(expected, percentiles)
    
    # De-duplicate thresholds in case of highly skewed data (e.g. Frequency with lots of 1s/2s)
    thresholds = np.unique(thresholds)
    if len(thresholds) < 2:
        return 0.0  # Can't compute bins if all values are identical

    # Adjust lower and upper bounds slightly to ensure all data falls within bin boundaries
    thresholds[0] -= 1e-5
    thresholds[-1] += 1e-5

    # Compute histograms
    expected_counts, _ = np.histogram(expected, bins=thresholds)
    actual_counts, _ = np.histogram(actual, bins=thresholds)

    # Convert counts to probabilities
    expected_probs = expected_counts / len(expected)
    actual_probs = actual_counts / len(actual)

    # Handle zero probability bins using a small epsilon
    eps = 1e-4
    expected_probs = np.where(expected_probs == 0, eps, expected_probs)
    actual_probs = np.where(actual_probs == 0, eps, actual_probs)

    # Re-normalize to sum to 1
    expected_probs /= expected_probs.sum()
    actual_probs /= actual_probs.sum()

    # Calculate PSI
    psi_value = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
    return float(psi_value)

def get_psi_status(psi_value: float) -> str:
    """
    Returns the drift status based on standard PSI thresholds.
    """
    if psi_value < 0.1:
        return "Stable"
    elif psi_value <= 0.25:
        return "Monitor"
    else:
        return "Retrain Alert"

def log_drift_to_db(db_path: str, drift_results: list):
    """
    Saves the data drift results to the Supabase/PostgreSQL database using SQLAlchemy.
    """
    # Ensure tables exist
    init_db()
    
    db = SessionLocal()
    timestamp = datetime.now().isoformat()
    try:
        for res in drift_results:
            metric = DriftMetric(
                timestamp=timestamp,
                feature_name=res['feature'],
                psi_value=res['psi_value'],
                status=res['status'],
                training_mean=res['training_mean'],
                production_mean=res['production_mean']
            )
            db.add(metric)
        db.commit()
        logger.info(f"Successfully logged drift metrics to Supabase/PostgreSQL database using SQLAlchemy")
    except Exception as e:
        logger.error(f"Failed logging drift metrics: {e}")
        db.rollback()
    finally:
        db.close()

def run_drift_detection(
    reference_path: str = "data/processed/rfm_features.csv",
    db_path: str = "backend/database.db",
    simulate_drift: bool = True
) -> dict:
    """
    Loads baseline RFM features, generates a simulated production batch, computes PSI,
    logs results to Supabase/PostgreSQL, and returns a JSON-serializable drift report.
    """
    logger.info("Starting Data Drift Detection pipeline.")
    
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference feature file not found at: {reference_path}")

    # Load reference dataset (training distribution)
    df_ref = pd.read_csv(reference_path)
    logger.info(f"Loaded reference dataset with {len(df_ref):,} rows.")

    # Select the core raw features to inspect for drift
    features_to_monitor = [
        'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
        'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
    ]
    
    # Simulate a "production" batch by sampling 20% of the dataset dynamically
    dynamic_seed = int(datetime.now().timestamp()) % 10000
    np.random.seed(dynamic_seed)
    df_prod = df_ref.sample(frac=0.2, random_state=dynamic_seed).copy()
    
    # If simulate_drift is active, apply a slight perturbation to mimic market shifts
    if simulate_drift:
        logger.info("Applying simulated drift perturbation to the production batch...")
        # Recency: slight multiplicative increase (customers purchasing marginally less recently)
        df_prod['Recency'] = df_prod['Recency'] * np.random.uniform(1.02, 1.08, size=len(df_prod))
        # Frequency: very slight decrease
        df_prod['Frequency'] = (df_prod['Frequency'] * np.random.uniform(0.92, 0.99, size=len(df_prod))).clip(lower=1).round()
        # Monetary: slight increase (inflation)
        df_prod['Monetary'] = df_prod['Monetary'] * np.random.uniform(1.01, 1.08, size=len(df_prod))
        # Recalculate dependent features
        df_prod['AvgOrderValue'] = df_prod['Monetary'] / df_prod['Frequency']
        df_prod['UniqueProducts'] = (df_prod['UniqueProducts'] * np.random.uniform(0.95, 1.05, size=len(df_prod))).clip(lower=1).round()
        df_prod['ReturnRate'] = (df_prod['ReturnRate'] + np.random.uniform(0.0, 0.02, size=len(df_prod))).clip(0, 1)
        # New features perturbations
        df_prod['CustomerLifetimeDays'] = (df_prod['CustomerLifetimeDays'] * np.random.uniform(0.98, 1.02, size=len(df_prod))).round()
        df_prod['PurchaseFrequencyMonthly'] = (df_prod['Frequency'] / (df_prod['CustomerLifetimeDays'] + 30)) * 30.0
        df_prod['AvgQuantityPerOrder'] = df_prod['AvgQuantityPerOrder'] * np.random.uniform(0.99, 1.01, size=len(df_prod))

    drift_results = []
    overall_status = "Stable"
    
    logger.info("Calculating PSI for features:")
    for feature in features_to_monitor:
        expected = df_ref[feature].values
        actual = df_prod[feature].values
        
        psi_val = calculate_psi(expected, actual, num_bins=10)
        status = get_psi_status(psi_val)
        
        # If any feature needs retraining, escalate overall status
        if status == "Retrain Alert":
            overall_status = "Retrain Alert"
        elif status == "Monitor" and overall_status == "Stable":
            overall_status = "Monitor"

        train_mean = float(expected.mean())
        prod_mean = float(actual.mean())
        
        drift_results.append({
            "feature": feature,
            "psi_value": psi_val,
            "status": status,
            "training_mean": train_mean,
            "production_mean": prod_mean
        })
        logger.info(f"Feature: {feature:15} | PSI: {psi_val:.4f} | Status: {status:13} | Train Mean: {train_mean:.2f} | Prod Mean: {prod_mean:.2f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "metrics": drift_results
    }

    # Log to Supabase/PostgreSQL
    log_drift_to_db(db_path, drift_results)
    
    # Save a JSON copy of the drift report in processed folder
    report_json_path = "data/processed/drift_report.json"
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved drift report JSON to {report_json_path}")

    return report

if __name__ == "__main__":
    run_drift_detection()
