"""
Test suite for ML pipeline components.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Prevent sklearn/joblib from trying to read Linux cgroup files on Windows hosts.
# Must be set BEFORE any sklearn imports happen (i.e., at module top-level).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

# Ensure project root is on the path so `backend.pipeline.*` imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPreprocessing:
    """Test suite for preprocessing pipeline."""

    def test_preprocessing_removes_null_customers(self):
        """Preprocessing should remove rows with null CustomerID."""
        from backend.pipeline.preprocessing import run_preprocessing

        test_data = pd.DataFrame({
            'CustomerID': [1.0, np.nan, 3.0],
            'InvoiceNo': ['A001', 'A002', 'A003'],
            'Quantity': [5, 10, 3],
            'UnitPrice': [2.0, 3.0, 1.5],
            'InvoiceDate': pd.date_range('2024-01-01', periods=3),
            'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom'],
            'Description': ['Item A', 'Item B', 'Item C'],
            'StockCode': ['S001', 'S002', 'S003']
        })

        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_excel', return_value=test_data), \
             patch('pandas.DataFrame.to_csv'), \
             patch('os.makedirs'):
            result = run_preprocessing(
                raw_path='mock.xlsx',
                output_path='mock_output.csv',
                filter_uk_only=False
            )
            # The null CustomerID row should have been removed
            assert len(result) == 2

    def test_preprocessing_removes_cancelled_invoices(self):
        """Preprocessing should filter out cancelled invoices (InvoiceNo starting with C)."""
        test_data = pd.DataFrame({
            'CustomerID': [1.0, 2.0, 3.0],
            'InvoiceNo': ['A001', 'C002', 'A003'],  # C002 is cancelled
            'Quantity': [5, 10, 3],
            'UnitPrice': [2.0, 3.0, 1.5],
            'InvoiceDate': pd.date_range('2024-01-01', periods=3),
            'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom'],
            'Description': ['Item A', 'Item B', 'Item C'],
            'StockCode': ['S001', 'S002', 'S003']
        })

        from backend.pipeline.preprocessing import run_preprocessing

        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_excel', return_value=test_data), \
             patch('pandas.DataFrame.to_csv'), \
             patch('os.makedirs'):
            result = run_preprocessing(
                raw_path='mock.xlsx',
                output_path='mock_output.csv',
                filter_uk_only=False
            )
            assert 'C002' not in result['InvoiceNo'].values


class TestRFMFeatures:
    """Test suite for RFM feature engineering."""

    def _make_cleaned_data(self):
        return pd.DataFrame({
            'CustomerID': ['1', '1', '2', '2', '2'],
            'InvoiceNo': ['A', 'B', 'C', 'D', 'E'],
            'InvoiceDate': pd.date_range('2024-01-01', periods=5),
            'TotalPrice': [100.0, 150.0, 50.0, 75.0, 100.0],
            'Quantity': [5, 10, 2, 3, 4],
            'StockCode': ['S1', 'S2', 'S1', 'S3', 'S2']
        })

    def _make_raw_data(self):
        return pd.DataFrame({
            'CustomerID': [1.0, 1.0, 2.0, 2.0, 2.0],
            'InvoiceNo': ['A', 'B', 'C', 'D', 'E']
        })

    def test_rfm_calculates_basic_metrics(self):
        """RFM engineering should calculate Recency, Frequency, Monetary and engineered features."""
        from backend.pipeline.rfm_features import run_rfm_engineering

        cleaned = self._make_cleaned_data()
        raw = self._make_raw_data()

        # raw_subset.csv is checked first; return False so it falls back to read_excel
        def exists_side_effect(path):
            if 'raw_subset' in str(path):
                return False
            return True  # cleaned_path and raw_path are "found"

        with patch('os.path.exists', side_effect=exists_side_effect), \
             patch('pandas.read_csv', return_value=cleaned), \
             patch('pandas.read_excel', return_value=raw), \
             patch('joblib.dump'), \
             patch('pandas.DataFrame.to_csv'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('seaborn.heatmap'), \
             patch('os.makedirs'):
            result = run_rfm_engineering(
                cleaned_path='mock_clean.csv',
                raw_path='mock_raw.xlsx',
                output_path='mock_rfm.csv',
                scaler_path='mock_scaler.pkl',
                correlation_plot_path='mock_corr.png'
            )

            assert 'Recency' in result.columns
            assert 'Frequency' in result.columns
            assert 'Monetary' in result.columns
            assert 'AvgOrderValue' in result.columns
            assert 'UniqueProducts' in result.columns
            assert 'ReturnRate' in result.columns

    def test_rfm_log_transformation(self):
        """RFM engineering should apply log1p transformation to Recency, Frequency, Monetary."""
        from backend.pipeline.rfm_features import run_rfm_engineering

        cleaned = self._make_cleaned_data()
        raw = self._make_raw_data()

        def exists_side_effect(path):
            if 'raw_subset' in str(path):
                return False
            return True

        with patch('os.path.exists', side_effect=exists_side_effect), \
             patch('pandas.read_csv', return_value=cleaned), \
             patch('pandas.read_excel', return_value=raw), \
             patch('joblib.dump'), \
             patch('pandas.DataFrame.to_csv'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('seaborn.heatmap'), \
             patch('os.makedirs'):
            result = run_rfm_engineering(
                cleaned_path='mock.csv',
                raw_path='mock.xlsx',
                output_path='mock_rfm.csv',
                scaler_path='mock_scaler.pkl',
                correlation_plot_path='mock_corr.png'
            )

            assert 'Recency_log' in result.columns
            assert 'Frequency_log' in result.columns
            assert 'Monetary_log' in result.columns


class TestSegmentation:
    """Test suite for customer segmentation."""

    def test_segmentation_assigns_clusters(self):
        """Segmentation should assign KMeans, DBSCAN, Hierarchical clusters and business Segment labels."""
        from backend.pipeline.segmentation import run_segmentation

        rfm_data = pd.DataFrame({
            'CustomerID': [str(i) for i in range(100)],
            'Recency_log_scaled': np.random.randn(100),
            'Frequency_log_scaled': np.random.randn(100),
            'Monetary_log_scaled': np.random.randn(100),
            'AvgOrderValue_scaled': np.random.randn(100),
            'UniqueProducts_scaled': np.random.randn(100),
            'ReturnRate_scaled': np.random.randn(100),
            'CustomerLifetimeDays_scaled': np.random.randn(100),
            'PurchaseFrequencyMonthly_scaled': np.random.randn(100),
            'AvgQuantityPerOrder_scaled': np.random.randn(100),
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 50, 100),
            'Monetary': np.random.rand(100) * 1000
        })

        mock_plt = MagicMock()
        mock_ax = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax.twinx.return_value = mock_ax2
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=rfm_data), \
             patch('pandas.DataFrame.to_csv'), \
             patch('backend.pipeline.segmentation.plt', mock_plt), \
             patch('backend.pipeline.segmentation.sns', MagicMock()), \
             patch('scipy.cluster.hierarchy.dendrogram'), \
             patch('scipy.cluster.hierarchy.linkage', return_value=np.zeros((99, 4))), \
             patch('os.makedirs'):
            result = run_segmentation(
                features_path='mock_rfm.csv',
                output_path='mock_segments.csv',
                plots_dir='mock_plots'
            )

            assert 'KMeans_Cluster' in result.columns
            assert 'DBSCAN_Cluster' in result.columns
            assert 'Hierarchical_Cluster' in result.columns
            assert 'Segment' in result.columns

            expected_segments = ['Champions', 'Loyal Customers', 'At-Risk', 'Lost Customers']
            for segment in expected_segments:
                assert segment in result['Segment'].unique()


class TestModelTraining:
    """Test suite for model training pipeline."""

    def test_model_training_saves_artifacts(self):
        """Model training should produce metrics for all 5 classifiers + baseline."""
        from backend.pipeline.model_training import run_model_training

        mock_df = pd.DataFrame({
            'CustomerID': [str(i) for i in range(200)],
            'Recency': np.random.randint(1, 365, 200),
            'Frequency': np.random.randint(1, 50, 200),
            'Monetary': np.random.rand(200) * 1000,
            'AvgOrderValue': np.random.rand(200) * 100,
            'UniqueProducts': np.random.randint(1, 20, 200),
            'ReturnRate': np.random.rand(200) * 0.3,
            'CustomerLifetimeDays': np.random.randint(30, 365, 200),
            'PurchaseFrequencyMonthly': np.random.rand(200) * 10,
            'AvgQuantityPerOrder': np.random.rand(200) * 20,
            'Target': np.random.randint(0, 2, 200)
        })

        mock_plt = MagicMock()
        mock_plt.figure.return_value = MagicMock()

        with patch('backend.pipeline.model_training.build_supervised_dataset', return_value=mock_df), \
             patch('backend.pipeline.model_training.plt', mock_plt), \
             patch('backend.pipeline.model_training.sns', MagicMock()), \
             patch('backend.pipeline.model_training.shap', MagicMock()), \
             patch('joblib.dump') as mock_dump, \
             patch('os.makedirs'), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'):
            metrics = run_model_training(
                cleaned_path='mock_clean.csv',
                raw_path='mock_raw.xlsx',
                models_dir='mock_models',
                plots_dir='mock_plots'
            )

            assert 'logistic_regression' in metrics
            assert 'random_forest' in metrics
            assert 'xgboost' in metrics
            assert 'lightgbm' in metrics
            assert 'stacking_ensemble' in metrics
            assert 'baseline_majority' in metrics
            assert mock_dump.called


class TestDriftDetection:
    """Test suite for drift detection."""

    def test_psi_calculation(self):
        """PSI calculation should return valid drift scores."""
        from backend.pipeline.drift_detection import calculate_psi

        np.random.seed(42)
        expected = np.random.normal(100, 20, 1000)
        actual_stable = np.random.normal(100, 20, 1000)   # No drift
        actual_drift = np.random.normal(120, 20, 1000)    # Significant drift

        psi_stable = calculate_psi(expected, actual_stable)
        psi_drift = calculate_psi(expected, actual_drift)

        assert psi_stable < 0.1
        assert psi_drift > psi_stable

    def test_psi_status_thresholds(self):
        """PSI status should correctly categorize drift levels."""
        from backend.pipeline.drift_detection import get_psi_status

        assert get_psi_status(0.05) == "Stable"
        assert get_psi_status(0.15) == "Monitor"
        assert get_psi_status(0.30) == "Retrain Alert"


# Fixtures for common test data
@pytest.fixture
def sample_customer_features():
    """Sample customer feature dictionary for predictions."""
    return {
        "Recency": 30.0,
        "Frequency": 10.0,
        "Monetary": 1000.0,
        "AvgOrderValue": 100.0,
        "UniqueProducts": 5.0,
        "ReturnRate": 0.1,
        "CustomerLifetimeDays": 200.0,
        "PurchaseFrequencyMonthly": 3.0,
        "AvgQuantityPerOrder": 15.0
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
