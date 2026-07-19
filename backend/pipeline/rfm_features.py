import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_rfm_engineering(
    cleaned_path: str = "data/processed/cleaned_retail.csv",
    raw_path: str = "data/raw/OnlineRetail.xlsx",
    output_path: str = "data/processed/rfm_features.csv",
    scaler_path: str = "models/scaler.pkl",
    correlation_plot_path: str = "data/plots/rfm_correlation.png"
):
    """
    Computes RFM features and engineered features for each customer, normalizes them, and saves the scaler/data.
    """
    logger.info("Starting RFM Feature Engineering pipeline.")
    
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned retail data not found at: {cleaned_path}")
    
    cached_raw_path = os.path.join(os.path.dirname(cleaned_path), "raw_subset.csv")
    if not os.path.exists(raw_path) and not os.path.exists(cached_raw_path):
        logger.info(f"Raw data file not found at: {raw_path} and cached subset missing. Attempting to download from UCI Repository...")
        import urllib.request
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, raw_path)
            logger.info("Successfully downloaded Online Retail dataset from UCI.")
        except Exception as download_err:
            raise FileNotFoundError(f"Raw data file not found at {raw_path} and failed to download from UCI: {download_err}")

    # Load cleaned dataset
    logger.info("Loading cleaned retail dataset...")
    df_cleaned = pd.read_csv(cleaned_path)
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(str)
    df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
    logger.info(f"Loaded {len(df_cleaned):,} rows of cleaned transactions.")

    # 1. Compute Return Rate using full dataset (before cancellation filter)
    logger.info("Computing customer ReturnRate from raw dataset...")
    cached_raw_path = os.path.join(os.path.dirname(cleaned_path), "raw_subset.csv")
    if os.path.exists(cached_raw_path):
        logger.info(f"Loading cached raw subset from {cached_raw_path}...")
        df_raw = pd.read_csv(cached_raw_path, usecols=['CustomerID', 'InvoiceNo'])
    else:
        logger.info("Cached raw subset not found. Reading from raw Excel (this will be slower)...")
        df_raw = pd.read_excel(raw_path, usecols=['CustomerID', 'InvoiceNo'])
    df_raw = df_raw.dropna(subset=['CustomerID'])
    df_raw['CustomerID'] = df_raw['CustomerID'].astype(float).astype(int).astype(str) # Handle float strings from CSV securely
    df_raw['InvoiceNo'] = df_raw['InvoiceNo'].astype(str).str.strip()

    # Calculate return rate per customer: count of unique 'C' invoices / total unique invoices
    logger.info("Grouping raw invoices to calculate ReturnRate...")
    customer_invoices = df_raw.groupby('CustomerID')['InvoiceNo'].nunique()
    customer_cancelled_invoices = df_raw[df_raw['InvoiceNo'].str.startswith('C')].groupby('CustomerID')['InvoiceNo'].nunique()
    
    # Fill missing values with 0 (customers without cancelled invoices)
    return_rates = (customer_cancelled_invoices / customer_invoices).fillna(0.0)
    df_returns = pd.DataFrame({'ReturnRate': return_rates}).reset_index()
    df_returns['CustomerID'] = df_returns['CustomerID'].astype(str)
    logger.info(f"Computed return rates for {len(df_returns):,} customers from raw data.")

    # 2. Compute RFM Features per customer on cleaned dataset
    # Recency: Days since last purchase (snapshot date = max InvoiceDate + 1 day)
    snapshot_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date for Recency: {snapshot_date}")

    logger.info("Computing Recency, Frequency, and Monetary features...")
    grouped = df_cleaned.groupby('CustomerID')
    recency = grouped['InvoiceDate'].max().apply(lambda x: (snapshot_date - x).days)
    first_purchase = grouped['InvoiceDate'].min()
    last_purchase = grouped['InvoiceDate'].max()
    frequency = grouped['InvoiceNo'].nunique()
    monetary = grouped['TotalPrice'].sum()
    total_quantity = grouped['Quantity'].sum()
    
    rfm = pd.DataFrame({
        'CustomerID': recency.index,
        'Recency': recency.values,
        'Frequency': frequency.values,
        'Monetary': monetary.values,
        'FirstPurchase': first_purchase.values,
        'LastPurchase': last_purchase.values,
        'TotalQuantity': total_quantity.values
    })

    # 3. Add extra engineered features
    logger.info("Adding extra engineered features (AvgOrderValue, UniqueProducts, CustomerLifetimeDays, PurchaseFrequencyMonthly, AvgQuantityPerOrder)...")
    # AvgOrderValue: Monetary / Frequency
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    
    # New features
    rfm['CustomerLifetimeDays'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days
    rfm['PurchaseFrequencyMonthly'] = (rfm['Frequency'] / (rfm['CustomerLifetimeDays'] + 30)) * 30.0
    rfm['AvgQuantityPerOrder'] = rfm['TotalQuantity'] / rfm['Frequency']
    
    # UniqueProducts: Count of distinct StockCodes purchased
    unique_products = df_cleaned.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_products['CustomerID'] = unique_products['CustomerID'].astype(str)
    unique_products = unique_products.rename(columns={'StockCode': 'UniqueProducts'})
    rfm = rfm.merge(unique_products, on='CustomerID', how='left')

    # ReturnRate (merge computed ReturnRate)
    rfm = rfm.merge(df_returns, on='CustomerID', how='left').fillna({'ReturnRate': 0.0})
    
    # Clean up temp columns
    rfm = rfm.drop(columns=['FirstPurchase', 'LastPurchase', 'TotalQuantity'])
    
    logger.info(f"Base RFM & Engineered features computed for {len(rfm):,} unique customers.")

    # 4. Apply log1p transformation to skewed features (Recency, Frequency, Monetary)
    logger.info("Applying log1p transformation to Recency, Frequency, and Monetary...")
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Also calculate skewness to log the metrics
    for col in ['Recency', 'Frequency', 'Monetary']:
        logger.info(f"Skewness of {col}: {rfm[col].skew():.2f} -> Log transformed: {rfm[f'{col}_log'].skew():.2f}")

    # 5. StandardScaler normalization
    feature_cols = [
        'Recency_log', 'Frequency_log', 'Monetary_log', 
        'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
        'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
    ]
    logger.info(f"Applying StandardScaler normalization on columns: {feature_cols}")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm[feature_cols])
    
    # Save the scaled features back to the dataframe
    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    df_scaled = pd.DataFrame(scaled_features, columns=scaled_cols, index=rfm.index)
    rfm = pd.concat([rfm, df_scaled], axis=1)

    # 6. Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved StandardScaler to {scaler_path}")

    # 7. Save final RFM feature matrix to /data/processed/rfm_features.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm.to_csv(output_path, index=False)
    logger.info(f"Saved RFM features to {output_path}")

    # 8. Output a feature correlation heatmap as /data/plots/rfm_correlation.png
    os.makedirs(os.path.dirname(correlation_plot_path), exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    # Correlation on core raw and engineered features
    corr_cols = [
        'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
        'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
    ]
    corr_matrix = rfm[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("RFM & Engineered Features Correlation Heatmap", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(correlation_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to {correlation_plot_path}")

    return rfm

if __name__ == "__main__":
    run_rfm_engineering()
