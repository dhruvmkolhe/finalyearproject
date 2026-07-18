"""
Comprehensive API test suite for PredictIQ backend endpoints.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Prevent sklearn/joblib from reading Linux cgroup files on Windows.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

# Ensure project root AND backend dir are on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import backend.main as main_module
from backend.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: inject fake scaler + models into backend.main globals so that
# prediction endpoints return 200 (not 503 "models not loaded").
# ---------------------------------------------------------------------------
def _mock_scaler():
    s = MagicMock()
    s.transform.return_value = np.zeros((1, 9))
    return s


def _mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([1])
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    return m


@pytest.fixture(autouse=False)
def inject_models():
    """Inject mock scaler and models into backend.main module globals."""
    original_scaler = main_module.scaler
    original_models = main_module.models.copy()

    main_module.scaler = _mock_scaler()
    main_module.models = {
        'logistic_regression': _mock_model(),
        'random_forest': _mock_model(),
        'xgboost': _mock_model(),
        'lightgbm': _mock_model(),
        'stacking_ensemble': _mock_model(),
    }
    # Disable xgb_shap_explainer to avoid SHAP calculation errors in mock
    main_module.xgb_shap_explainer = None

    yield

    main_module.scaler = original_scaler
    main_module.models = original_models


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/api/health")
        assert response.status_code == 200
    
    def test_health_endpoint_structure(self):
        """Health endpoint should return expected JSON structure."""
        response = client.get("/api/health")
        data = response.json()
        
        assert "success" in data
        assert "data" in data
        assert "error" in data
        assert data["success"] is True
    
    def test_health_endpoint_models_status(self):
        """Health endpoint should report model loading status."""
        response = client.get("/api/health")
        data = response.json()
        
        assert "models_loaded" in data["data"]
        models_status = data["data"]["models_loaded"]
        
        # Check all expected models
        expected_models = [
            'logistic_regression', 'random_forest', 'xgboost', 
            'lightgbm', 'stacking_ensemble', 'scaler'
        ]
        for model in expected_models:
            assert model in models_status


class TestDatasetStatsEndpoint:
    """Test suite for dataset statistics endpoint."""
    
    def test_dataset_stats_returns_200(self):
        """Dataset stats endpoint should return 200 OK."""
        response = client.get("/api/dataset/stats")
        assert response.status_code == 200
    
    def test_dataset_stats_structure(self):
        """Dataset stats should contain expected fields."""
        response = client.get("/api/dataset/stats")
        data = response.json()
        
        if data["success"]:
            stats = data["data"]
            assert "total_records" in stats
            assert "total_customers" in stats
            assert "total_transactions" in stats
            assert "date_range" in stats


class TestSegmentsEndpoint:
    """Test suite for customer segments endpoints."""
    
    def test_segments_overview_returns_200(self):
        """Segments overview endpoint should return 200 OK."""
        response = client.get("/api/segments/overview")
        assert response.status_code == 200
    
    def test_segments_overview_structure(self):
        """Segments overview should contain distribution and centroids."""
        response = client.get("/api/segments/overview")
        data = response.json()
        
        if data["success"]:
            overview = data["data"]
            assert "distribution" in overview
            assert "centroids" in overview
    
    def test_segments_customers_returns_200(self):
        """Segments customers endpoint should return 200 OK."""
        response = client.get("/api/segments/customers")
        assert response.status_code == 200


class TestModelsEndpoint:
    """Test suite for model metrics endpoints."""
    
    def test_models_metrics_returns_200(self):
        """Models metrics endpoint should return 200 OK."""
        response = client.get("/api/models/metrics")
        assert response.status_code == 200
    
    def test_models_metrics_structure(self):
        """Models metrics should contain all classifiers."""
        response = client.get("/api/models/metrics")
        data = response.json()
        
        if data["success"]:
            metrics = data["data"]["metrics"]
            expected_models = [
                'logistic_regression', 'random_forest', 'xgboost',
                'lightgbm', 'stacking_ensemble', 'baseline_majority'
            ]
            for model in expected_models:
                assert model in metrics
    
    def test_models_metrics_values(self):
        """Models metrics should contain expected performance metrics."""
        response = client.get("/api/models/metrics")
        data = response.json()
        
        if data["success"]:
            for model_name, model_metrics in data["data"]["metrics"].items():
                # Check required metrics exist
                required_metrics = [
                    'accuracy', 'precision', 'recall', 'f1_score',
                    'roc_auc', 'mcc'
                ]
                for metric in required_metrics:
                    assert metric in model_metrics
                    # Metrics should be between 0 and 1
                    assert 0 <= model_metrics[metric] <= 1


class TestPredictionEndpoint:
    """Test suite for single prediction endpoint."""

    @pytest.fixture(autouse=True)
    def setup_models(self, inject_models):
        """Auto-use inject_models for every test in this class."""
        pass

    def test_predict_single_with_valid_input(self):
        """Single prediction should work with valid customer features."""
        payload = {
            "Recency": 10.0,
            "Frequency": 5.0,
            "Monetary": 500.0,
            "AvgOrderValue": 100.0,
            "UniqueProducts": 3.0,
            "ReturnRate": 0.1,
            "CustomerLifetimeDays": 180.0,
            "PurchaseFrequencyMonthly": 2.5,
            "AvgQuantityPerOrder": 10.0
        }

        response = client.post("/api/predict/single", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
        assert "assigned_segment" in data["data"]

    def test_predict_single_with_missing_fields(self):
        """Single prediction should fail with missing required fields."""
        payload = {
            "Recency": 10.0,
            "Frequency": 5.0
            # Missing required fields
        }

        response = client.post("/api/predict/single", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_single_with_invalid_types(self):
        """Single prediction should fail with invalid data types."""
        payload = {
            "Recency": "invalid",  # Should be float
            "Frequency": 5.0,
            "Monetary": 500.0,
            "AvgOrderValue": 100.0,
            "UniqueProducts": 3.0,
            "ReturnRate": 0.1,
            "CustomerLifetimeDays": 180.0,
            "PurchaseFrequencyMonthly": 2.5,
            "AvgQuantityPerOrder": 10.0
        }

        response = client.post("/api/predict/single", json=payload)
        assert response.status_code == 422


class TestDriftEndpoint:
    """Test suite for drift detection endpoints."""
    
    def test_drift_status_returns_200(self):
        """Drift status endpoint should return 200 OK."""
        response = client.get("/api/drift/status")
        assert response.status_code == 200
    
    def test_drift_status_structure(self):
        """Drift status should contain metrics and overall status."""
        response = client.get("/api/drift/status")
        data = response.json()
        
        if data["success"]:
            drift_data = data["data"]
            assert "overall_status" in drift_data
            assert drift_data["overall_status"] in ["Stable", "Monitor", "Retrain Alert"]


class TestChartsEndpoint:
    """Test suite for visualization endpoints."""
    
    def test_charts_endpoint_valid_chart(self):
        """Charts endpoint should return image for valid chart name."""
        valid_charts = ["roc", "pr", "shap_summary", "pca"]
        
        for chart_name in valid_charts:
            response = client.get(f"/api/charts/{chart_name}")
            # Should either return 200 (image found) or 404 (not generated yet)
            assert response.status_code in [200, 404]
    
    def test_charts_endpoint_invalid_chart(self):
        """Charts endpoint should return 404 for invalid chart name."""
        response = client.get("/api/charts/invalid_chart_name")
        assert response.status_code == 404


class TestRetrainingEndpoint:
    """Test suite for model retraining endpoints."""
    
    def test_retrain_status_returns_200(self):
        """Retrain status endpoint should return 200 OK."""
        response = client.get("/api/models/retrain/status")
        assert response.status_code == 200
    
    def test_retrain_status_structure(self):
        """Retrain status should contain expected fields."""
        response = client.get("/api/models/retrain/status")
        data = response.json()
        
        assert data["success"] is True
        status_data = data["data"]
        
        assert "status" in status_data
        assert "progress" in status_data
        assert status_data["status"] in ["idle", "running", "success", "failed"]
        assert 0 <= status_data["progress"] <= 100


@pytest.mark.asyncio
class TestChatEndpoint:
    """Test suite for AI chat endpoint."""
    
    async def test_chat_endpoint_without_api_key(self):
        """Chat endpoint should handle missing API key gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            payload = {
                "message": "What is the dataset size?",
                "history": []
            }
            
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Should return error about missing API key
            if not data["success"]:
                assert "GROQ_API_KEY" in data["error"]


# Performance and Load Tests
class TestPerformance:
    """Basic performance tests."""
    
    def test_health_endpoint_response_time(self):
        """Health endpoint should respond quickly."""
        import time
        start = time.time()
        response = client.get("/api/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Should respond in under 1 second


# Integration Tests
class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @pytest.fixture(autouse=True)
    def setup_models(self, inject_models):
        """Auto-use inject_models for every test in this class."""
        pass

    def test_full_prediction_workflow(self):
        """Test complete prediction workflow from input to output."""
        # 1. Check system health
        health = client.get("/api/health")
        assert health.status_code == 200

        # 2. Get model metrics
        metrics = client.get("/api/models/metrics")
        assert metrics.status_code == 200

        # 3. Make a prediction
        payload = {
            "Recency": 15.0,
            "Frequency": 8.0,
            "Monetary": 800.0,
            "AvgOrderValue": 100.0,
            "UniqueProducts": 5.0,
            "ReturnRate": 0.05,
            "CustomerLifetimeDays": 200.0,
            "PurchaseFrequencyMonthly": 3.0,
            "AvgQuantityPerOrder": 12.0
        }

        prediction = client.post("/api/predict/single", json=payload)
        assert prediction.status_code == 200

        # 4. Check prediction history
        history = client.get("/api/predict/history")
        assert history.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

