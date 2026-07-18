import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline_runner")

def run_all_phases():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("           PredictIQ Machine Learning Pipeline Runner            ")
    logger.info("=================================================================")
    
    # Add root folder to python path for module imports
    sys.path.append(os.getcwd())
    
    try:
        # Phase 1: Preprocessing
        logger.info("\n--- Phase 1: Running Preprocessing ---")
        from backend.pipeline.preprocessing import run_preprocessing
        run_preprocessing(
            raw_path="data/raw/OnlineRetail.xlsx",
            output_path="data/processed/cleaned_retail.csv",
            filter_uk_only=True
        )
        logger.info("Phase 1 completed successfully.")
        
        # Phase 2: RFM Feature Engineering
        logger.info("\n--- Phase 2: Running RFM Feature Engineering ---")
        from backend.pipeline.rfm_features import run_rfm_engineering
        run_rfm_engineering(
            cleaned_path="data/processed/cleaned_retail.csv",
            raw_path="data/raw/OnlineRetail.xlsx",
            output_path="data/processed/rfm_features.csv",
            scaler_path="models/scaler.pkl",
            correlation_plot_path="data/plots/rfm_correlation.png"
        )
        logger.info("Phase 2 completed successfully.")
        
        # Phase 3: Customer Segmentation
        logger.info("\n--- Phase 3: Running Unsupervised Segmentation ---")
        from backend.pipeline.segmentation import run_segmentation
        run_segmentation(
            features_path="data/processed/rfm_features.csv",
            output_path="data/processed/segmented_customers.csv",
            plots_dir="data/plots"
        )
        logger.info("Phase 3 completed successfully.")
        
        # Phase 4: Model Training
        logger.info("\n--- Phase 4: Running Supervised Classifier Training ---")
        from backend.pipeline.model_training import run_model_training
        run_model_training(
            cleaned_path="data/processed/cleaned_retail.csv",
            raw_path="data/raw/OnlineRetail.xlsx",
            models_dir="models",
            plots_dir="data/plots"
        )
        logger.info("Phase 4 completed successfully.")
        
        # Phase 5: Drift Detection
        logger.info("\n--- Phase 5: Running Data Drift Detection ---")
        from backend.pipeline.drift_detection import run_drift_detection
        run_drift_detection(
            reference_path="data/processed/rfm_features.csv",
            db_path="backend/database.db",
            simulate_drift=True
        )
        logger.info("Phase 5 completed successfully.")
        
        elapsed = time.time() - start_time
        logger.info("=================================================================")
        logger.info(f"Pipeline executed successfully in {elapsed:.2f} seconds.")
        logger.info("All artifacts and model packages are up to date.")
        logger.info("=================================================================")
        
    except Exception as e:
        logger.error(f"Pipeline failed at phase execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_all_phases()
