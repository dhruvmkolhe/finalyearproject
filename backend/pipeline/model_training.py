import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix,
    matthews_corrcoef
)
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_supervised_dataset(
    cleaned_path: str = "data/processed/cleaned_retail.csv",
    raw_path: str = "data/raw/OnlineRetail.xlsx"
):
    """
    Creates feature-target split based on 9-month/3-month timeline.
    Features: first 9 months
    Target: last 3 months (repurchase within next 30 days / next period)
    """
    logger.info("Building supervised dataset from retail timeline.")
    df_cleaned = pd.read_csv(cleaned_path)
    df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(str)

    min_date = df_cleaned['InvoiceDate'].min()
    split_date = min_date + pd.DateOffset(months=9)
    logger.info(f"Timeline: Start={min_date} | Split Date={split_date} | End={df_cleaned['InvoiceDate'].max()}")

    # 1. Feature Period (first 9 months)
    df_feats_raw = df_cleaned[df_cleaned['InvoiceDate'] < split_date].copy()
    logger.info(f"Feature period transactions: {len(df_feats_raw):,} rows.")

    # 2. Target Period (last 3+ months)
    df_target_raw = df_cleaned[df_cleaned['InvoiceDate'] >= split_date].copy()
    logger.info(f"Target period transactions: {len(df_target_raw):,} rows.")

    # Get active customers in the first 9 months
    active_customers = df_feats_raw['CustomerID'].unique()
    logger.info(f"Found {len(active_customers):,} unique active customers in feature period.")

    # Recency, Frequency, Monetary in feature period
    logger.info("Computing RFM features on first 9 months...")
    grouped = df_feats_raw.groupby('CustomerID')
    recency = grouped['InvoiceDate'].max().apply(lambda x: (split_date - x).days)
    first_purchase = grouped['InvoiceDate'].min()
    last_purchase = grouped['InvoiceDate'].max()
    frequency = grouped['InvoiceNo'].nunique()
    monetary = grouped['TotalPrice'].sum()
    total_quantity = grouped['Quantity'].sum()
    
    rfm_feats = pd.DataFrame({
        'CustomerID': recency.index,
        'Recency': recency.values,
        'Frequency': frequency.values,
        'Monetary': monetary.values,
        'FirstPurchase': first_purchase.values,
        'LastPurchase': last_purchase.values,
        'TotalQuantity': total_quantity.values
    })

    # Extra features in feature period
    logger.info("Adding engineered features...")
    rfm_feats['AvgOrderValue'] = rfm_feats['Monetary'] / rfm_feats['Frequency']
    rfm_feats['CustomerLifetimeDays'] = (rfm_feats['LastPurchase'] - rfm_feats['FirstPurchase']).dt.days
    rfm_feats['PurchaseFrequencyMonthly'] = (rfm_feats['Frequency'] / (rfm_feats['CustomerLifetimeDays'] + 30)) * 30.0
    rfm_feats['AvgQuantityPerOrder'] = rfm_feats['TotalQuantity'] / rfm_feats['Frequency']
    
    unique_prods = df_feats_raw.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_prods = unique_prods.rename(columns={'StockCode': 'UniqueProducts'})
    rfm_feats = rfm_feats.merge(unique_prods, on='CustomerID', how='left')

    # Drop temporary columns
    rfm_feats = rfm_feats.drop(columns=['FirstPurchase', 'LastPurchase', 'TotalQuantity'])

    # Return rate in feature period from raw data
    logger.info("Computing ReturnRate from raw data for feature period...")
    cached_raw_path = os.path.join(os.path.dirname(cleaned_path), "raw_subset.csv")
    if os.path.exists(cached_raw_path):
        logger.info(f"Loading cached raw subset from {cached_raw_path}...")
        df_raw = pd.read_csv(cached_raw_path, usecols=['CustomerID', 'InvoiceNo', 'InvoiceDate'])
    else:
        logger.info("Cached raw subset not found. Reading from raw Excel (this will be slower)...")
        if not os.path.exists(raw_path):
            logger.info(f"Raw data file not found at: {raw_path}. Attempting to download from UCI Repository...")
            import urllib.request
            raw_dir = os.path.dirname(raw_path)
            if raw_dir:
                os.makedirs(raw_dir, exist_ok=True)
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(url, raw_path)
                logger.info("Successfully downloaded Online Retail dataset from UCI.")
            except Exception as download_err:
                raise FileNotFoundError(f"Raw data file not found at {raw_path} and failed to download from UCI: {download_err}")
        df_raw = pd.read_excel(raw_path, usecols=['CustomerID', 'InvoiceNo', 'InvoiceDate'])
    df_raw = df_raw.dropna(subset=['CustomerID'])
    df_raw['CustomerID'] = df_raw['CustomerID'].astype(float).astype(int).astype(str) # Handle float float-like representations from CSV safely
    df_raw['InvoiceNo'] = df_raw['InvoiceNo'].astype(str).str.strip()
    df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
    
    # Filter raw to feature period
    df_raw_feats = df_raw[df_raw['InvoiceDate'] < split_date]
    customer_invoices = df_raw_feats.groupby('CustomerID')['InvoiceNo'].nunique()
    customer_cancelled_invoices = df_raw_feats[df_raw_feats['InvoiceNo'].str.startswith('C')].groupby('CustomerID')['InvoiceNo'].nunique()
    
    return_rates = (customer_cancelled_invoices / customer_invoices).fillna(0.0)
    df_returns = pd.DataFrame({'ReturnRate': return_rates}).reset_index()
    df_returns['CustomerID'] = df_returns['CustomerID'].astype(str)
    
    rfm_feats = rfm_feats.merge(df_returns, on='CustomerID', how='left').fillna({'ReturnRate': 0.0})

    # Define target label y: did they purchase in target period?
    target_buyers = set(df_target_raw['CustomerID'].unique())
    rfm_feats['Target'] = rfm_feats['CustomerID'].apply(lambda x: 1 if x in target_buyers else 0)

    logger.info(f"Target distribution: {rfm_feats['Target'].value_counts().to_dict()}")

    return rfm_feats

def run_model_training(
    cleaned_path: str = "data/processed/cleaned_retail.csv",
    raw_path: str = "data/raw/OnlineRetail.xlsx",
    models_dir: str = "models",
    plots_dir: str = "data/plots"
):
    """
    Fits and evaluates 5 classification models, saves models and metrics, generates evaluation plots.
    """
    logger.info("Starting Supervised Model Training pipeline.")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Build supervised features & target
    data = build_supervised_dataset(cleaned_path, raw_path)
    
    # Save the supervised dataset to disk for the real-time WebSocket simulator
    sup_csv_path = os.path.join(os.path.dirname(cleaned_path), "supervised_data.csv")
    data.to_csv(sup_csv_path, index=False)
    logger.info(f"Saved updated 9D supervised dataset to {sup_csv_path}")
    
    # 2. Skewness correction
    data['Recency_log'] = np.log1p(data['Recency'])
    data['Frequency_log'] = np.log1p(data['Frequency'])
    data['Monetary_log'] = np.log1p(data['Monetary'])

    feature_cols = [
        'Recency_log', 'Frequency_log', 'Monetary_log', 
        'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
        'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
    ]
    X_df = data[feature_cols]
    y = data['Target'].values

    # 3. Train-test split (80/20 stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {len(X_train_raw):,} | Test set: {len(X_test_raw):,}")

    # 4. Fit & save scaler on train set features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test_raw.index)

    # Save Scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved fitted StandardScaler to {scaler_path}")

    # 5. Handle Class Imbalance using SMOTE
    logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")

    # 6. Hyperparameter Tuning using RandomizedSearchCV
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    # 6a. Random Forest Tuning
    logger.info("Tuning Random Forest hyperparameters...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42
    )
    rf_search.fit(X_train_res, y_train_res)
    rf_model = rf_search.best_estimator_
    logger.info(f"RF Best Params: {rf_search.best_params_}")

    # 6b. XGBoost Tuning
    logger.info("Tuning XGBoost hyperparameters...")
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
        param_distributions=xgb_param_grid,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42
    )
    xgb_search.fit(X_train_res, y_train_res)
    xgb_model = xgb_search.best_estimator_
    logger.info(f"XGB Best Params: {xgb_search.best_params_}")

    # 6c. LightGBM Tuning
    logger.info("Tuning LightGBM hyperparameters...")
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    lgb_search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, verbose=-1),
        param_distributions=lgb_param_grid,
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42
    )
    lgb_search.fit(X_train_res, y_train_res)
    lgb_model = lgb_search.best_estimator_
    logger.info(f"LGB Best Params: {lgb_search.best_params_}")

    # 6d. Base Models for Stacking Ensemble (including base LR and GB for diversity)
    lr_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    # Replace slow SVC with a fast base Logistic Regression classifier for stacking ensemble
    svc_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42) 
    gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)

    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('svc', svc_model), # Maintain key 'svc' to prevent downstream breakage, using fast LR
        ('gb', gb_model)
    ]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=3,
        stack_method='predict_proba',
        n_jobs=1
    )

    models = {
        'logistic_regression': lr_model,
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'stacking_ensemble': stacking_model
    }

    metrics_dict = {}

    plt.figure(figsize=(10, 8))
    plt.title("ROC Curves Comparison", fontsize=14, fontweight='bold', pad=15)
    
    plt_pr = plt.figure(figsize=(10, 8))
    plt.figure(plt_pr.number)
    plt.title("Precision-Recall Curves Comparison", fontsize=14, fontweight='bold', pad=15)

    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        # Train on SMOTE resampled data
        model.fit(X_train_res, y_train_res)
        
        # Save model
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")

        # Predict
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall_curve, precision_curve)

        # Cross Validation (3-fold) using ROC-AUC on the resampled training set for fast evaluation
        logger.info(f"Running 3-fold CV for {model_name}...")
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=3, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        metrics_dict[model_name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "mcc": float(mcc),
            "cv_roc_auc_mean": float(cv_mean),
            "cv_roc_auc_std": float(cv_std)
        }
        logger.info(f"{model_name} Metrics: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f} | MCC={mcc:.4f} | CV={cv_mean:.4f}±{cv_std:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Purchase', 'Purchase'],
                    yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontsize=12, pad=10)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Plot ROC overlay
        plt.figure(1) # ROC Comparison Figure
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})', linewidth=2)

        # Plot PR overlay
        plt.figure(plt_pr.number) # PR Comparison Figure
        plt.plot(recall_curve, precision_curve, label=f'{model_name.replace("_", " ").title()} (AUC = {pr_auc:.3f})', linewidth=2)

    # Train baseline Dummy Classifier (Majority Class)
    logger.info("Training baseline Dummy Classifier...")
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_train, y_train) # Fit on raw training set to represent actual majority baseline
    
    y_pred_dummy = dummy_model.predict(X_test)
    y_probs_dummy = dummy_model.predict_proba(X_test)[:, 1]
    
    acc_dummy = accuracy_score(y_test, y_pred_dummy)
    prec_dummy = precision_score(y_test, y_pred_dummy, zero_division=0)
    rec_dummy = recall_score(y_test, y_pred_dummy, zero_division=0)
    f1_dummy = f1_score(y_test, y_pred_dummy, zero_division=0)
    roc_auc_dummy = roc_auc_score(y_test, y_probs_dummy)
    mcc_dummy = matthews_corrcoef(y_test, y_pred_dummy)
    
    precision_curve_dummy, recall_curve_dummy, _ = precision_recall_curve(y_test, y_probs_dummy)
    pr_auc_dummy = auc(recall_curve_dummy, precision_curve_dummy)
    
    metrics_dict['baseline_majority'] = {
        "accuracy": float(acc_dummy),
        "precision": float(prec_dummy),
        "recall": float(rec_dummy),
        "f1_score": float(f1_dummy),
        "roc_auc": float(roc_auc_dummy),
        "pr_auc": float(pr_auc_dummy),
        "mcc": float(mcc_dummy),
        "cv_roc_auc_mean": 0.5,
        "cv_roc_auc_std": 0.0
    }
    logger.info(f"Baseline (Majority Class) Metrics: Acc={acc_dummy:.4f} | Prec={prec_dummy:.4f} | Rec={rec_dummy:.4f} | F1={f1_dummy:.4f} | ROC-AUC={roc_auc_dummy:.4f} | MCC={mcc_dummy:.4f}")

    # Save ROC curves comparison
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_plot_path = os.path.join(plots_dir, "roc_curve_comparison.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC comparison plot to {roc_plot_path}")

    # Save PR curves comparison
    plt.figure(plt_pr.number)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    pr_plot_path = os.path.join(plots_dir, "pr_curve_comparison.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR comparison plot to {pr_plot_path}")

    # 7. Plot Random Forest Feature Importance
    logger.info("Extracting and saving Random Forest Feature Importances...")
    rf_importances = rf_model.feature_importances_
    indices = np.argsort(rf_importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rf_importances[indices], y=[feature_cols[i] for i in indices], palette='viridis')
    plt.title('Random Forest Feature Importances', fontsize=14, pad=15)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    rf_fi_plot = os.path.join(plots_dir, "rf_feature_importances.png")
    plt.savefig(rf_fi_plot, dpi=300)
    plt.close()
    logger.info(f"Saved RF feature importances to {rf_fi_plot}")

    # 8. SHAP Explainability on XGBoost model
    logger.info("Running SHAP explainability on XGBoost model...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    # Handle binary shape differences across SHAP versions
    if len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    # Beeswarm summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test, show=False)
    plt.title("SHAP Beeswarm Summary Plot (XGBoost)", fontsize=14, pad=15)
    plt.tight_layout()
    shap_summary_plot_path = os.path.join(plots_dir, "shap_summary.png")
    plt.savefig(shap_summary_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP summary beeswarm plot to {shap_summary_plot_path}")

    # Bar plot (Mean |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_to_plot, show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, pad=15)
    plt.tight_layout()
    shap_bar_plot_path = os.path.join(plots_dir, "shap_bar.png")
    plt.savefig(shap_bar_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP bar plot to {shap_bar_plot_path}")

    # 9. Save metrics to JSON
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved model metrics JSON to {metrics_path}")

    return metrics_dict

if __name__ == "__main__":
    run_model_training()
