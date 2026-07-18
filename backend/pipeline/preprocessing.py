import os
import logging
import pandas as pd
import numpy as np

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_preprocessing(
    raw_path: str = "data/raw/OnlineRetail.xlsx",
    output_path: str = "data/processed/cleaned_retail.csv",
    filter_uk_only: bool = True
):
    """
    Load raw online retail dataset, clean it, extract features, and save processed data.
    """
    logger.info(f"Starting preprocessing pipeline. Raw data source: {raw_path}")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    # Step 1: Load dataset
    logger.info("Loading Excel dataset (this might take a minute)...")
    df = pd.read_excel(raw_path)
    initial_rows = len(df)
    logger.info(f"Loaded raw dataset with {initial_rows:,} rows and {len(df.columns)} columns.")
    
    # Save a subset of raw columns to avoid re-reading Excel in subsequent phases
    try:
        raw_subset_path = os.path.join(os.path.dirname(output_path), "raw_subset.csv")
        os.makedirs(os.path.dirname(raw_subset_path), exist_ok=True)
        df[['CustomerID', 'InvoiceNo', 'InvoiceDate']].to_csv(raw_subset_path, index=False)
        logger.info(f"Saved raw subset to {raw_subset_path} for caching downstream.")
    except Exception as cache_err:
        logger.warning(f"Could not cache raw subset: {cache_err}")

    # Log column names and types
    logger.info(f"Columns: {list(df.columns)}")

    # Step 2: Drop rows where CustomerID is null
    prev_rows = len(df)
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str) # Normalize CustomerID to string representation of int
    logger.info(f"Dropped rows with null CustomerID. Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")

    # Step 3: Remove cancelled invoices (InvoiceNo starting with 'C')
    prev_rows = len(df)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    logger.info(f"Removed cancelled invoices (InvoiceNo starting with 'C'). Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")

    # Step 4: Remove rows where Quantity <= 0 or UnitPrice <= 0
    prev_rows = len(df)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    logger.info(f"Removed rows with Quantity <= 0 or UnitPrice <= 0. Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")

    # Step 5: Parse InvoiceDate as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    logger.info("Parsed InvoiceDate as datetime successfully.")

    # Step 6: Add derived column: TotalPrice = Quantity × UnitPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    logger.info("Created derived column 'TotalPrice' = Quantity * UnitPrice.")

    # Step 7: Filter to UK customers only (Country == 'United Kingdom') for cleaner segmentation (Toggable)
    if filter_uk_only:
        prev_rows = len(df)
        df = df[df['Country'] == 'United Kingdom']
        logger.info(f"Filtered to UK customers only. Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")
    else:
        logger.info("UK customer filtering is disabled; keeping all countries.")

    # Step 8: Remove obvious outliers using IQR method on TotalPrice and Quantity
    # Pre-outlier check metrics
    logger.info(f"Summary statistics before outlier removal:\n{df[['Quantity', 'TotalPrice']].describe().to_string()}")

    # Outliers in Quantity
    prev_rows = len(df)
    q1_q = df['Quantity'].quantile(0.25)
    q3_q = df['Quantity'].quantile(0.75)
    iqr_q = q3_q - q1_q
    lower_q = q1_q - 1.5 * iqr_q
    upper_q = q3_q + 1.5 * iqr_q
    df = df[(df['Quantity'] >= lower_q) & (df['Quantity'] <= upper_q)]
    logger.info(f"Removed Quantity outliers (IQR: {iqr_q}, Bounds: [{lower_q}, {upper_q}]). Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")

    # Outliers in TotalPrice
    prev_rows = len(df)
    q1_tp = df['TotalPrice'].quantile(0.25)
    q3_tp = df['TotalPrice'].quantile(0.75)
    iqr_tp = q3_tp - q1_tp
    lower_tp = q1_tp - 1.5 * iqr_tp
    upper_tp = q3_tp + 1.5 * iqr_tp
    df = df[(df['TotalPrice'] >= lower_tp) & (df['TotalPrice'] <= upper_tp)]
    logger.info(f"Removed TotalPrice outliers (IQR: {iqr_tp}, Bounds: [{lower_tp}, {upper_tp}]). Rows: {prev_rows:,} -> {len(df):,} (Removed {(prev_rows - len(df)):,} rows)")

    logger.info(f"Summary statistics after outlier removal:\n{df[['Quantity', 'TotalPrice']].describe().to_string()}")

    # Step 9: Save cleaned dataset to /data/processed/cleaned_retail.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset of {len(df):,} rows to {output_path}")
    logger.info(f"Overall data retention: {len(df) / initial_rows * 100:.2f}% of raw rows retained ({len(df):,} out of {initial_rows:,} rows)")
    
    return df

if __name__ == "__main__":
    run_preprocessing()
