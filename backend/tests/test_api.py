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
    original_centroids = getattr(main_module, 'db_centroids', None)

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
    main_module.db_centroids = {"Champions": np.zeros(9)}

    yield

    main_module.scaler = original_scaler
    main_module.models = original_models
    main_module.db_centroids = original_centroids


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


class TestExtraEndpointsSuccess:
    """Test suite for additional backend endpoints to gain high coverage."""
    
    @pytest.fixture(autouse=True)
    def setup_models(self, inject_models):
        """Auto-use inject_models to avoid 503 errors."""
        pass

    def test_predict_batch_success(self):
        """Batch prediction returns processed CSV file successfully."""
        import io
        csv_data = (
            "Recency,Frequency,Monetary,AvgOrderValue,UniqueProducts,ReturnRate,"
            "CustomerLifetimeDays,PurchaseFrequencyMonthly,AvgQuantityPerOrder\n"
            "30.0,5.0,500.0,100.0,3.0,0.1,180.0,2.5,10.0\n"
        )
        files = {"file": ("test.csv", csv_data, "text/csv")}
        response = client.post("/api/predict/batch", files=files)
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "Predicted_Segment" in response.text
        assert "Predicted_Purchase_Label" in response.text
        assert "Purchase_Probability" in response.text

    def test_predict_batch_missing_columns(self):
        """Batch prediction fails if required columns are missing."""
        csv_data = "Recency,Frequency,Monetary\n30.0,5.0,500.0\n"
        files = {"file": ("test.csv", csv_data, "text/csv")}
        response = client.post("/api/predict/batch", files=files)
        assert response.status_code == 400
        assert "missing required feature columns" in response.json()["detail"]

    def test_trigger_drift_check(self):
        """POST /api/drift/check triggers the drift detection pipeline."""
        mock_report = {"overall_status": "Stable", "metrics": []}
        with patch('backend.pipeline.drift_detection.run_drift_detection', return_value=mock_report):
            response = client.post("/api/drift/check")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"] == mock_report

    def test_trigger_drift_check_failure(self):
        """POST /api/drift/check handles exceptions gracefully."""
        with patch('backend.pipeline.drift_detection.run_drift_detection', side_effect=Exception("Drift error")):
            response = client.post("/api/drift/check")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Drift error" in data["error"]

    def test_post_retrain_triggers_task(self):
        """POST /api/models/retrain triggers retraining background task."""
        with patch('backend.main.execute_pipeline_retraining') as mock_execute_retrain:
            response = client.post("/api/models/retrain")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "launched in background" in data["message"]

    def test_get_diagram_png_success(self):
        """GET /api/diagrams/png/{name} returns FileResponse."""
        with patch('backend.main.get_mermaid_png_path', return_value='mock_image.png'), \
             patch('backend.main.FileResponse') as mock_fileresponse:
            mock_fileresponse.return_value = "file_response_object"
            response = client.get("/api/diagrams/png/sys_arch")
            assert response.status_code == 200

    def test_get_diagram_png_failure(self):
        """GET /api/diagrams/png/{name} handles error."""
        with patch('backend.main.get_mermaid_png_path', side_effect=ValueError("Unknown diagram")):
            response = client.get("/api/diagrams/png/invalid_name")
            assert response.status_code == 500

    def test_download_diagram_png_success(self):
        """GET /api/diagrams/download-png/{name} sets correct headers."""
        with patch('backend.main.get_mermaid_png_path', return_value='mock_image.png'), \
             patch('backend.main.FileResponse') as mock_fileresponse:
            mock_fileresponse.return_value = "file_response_object"
            response = client.get("/api/diagrams/download-png/sys_arch")
            assert response.status_code == 200

    def test_get_diagrams_html_page_success(self):
        """GET /diagrams serves the HTML page if it exists."""
        with patch('os.path.exists', return_value=True), \
             patch('backend.main.FileResponse') as mock_fileresponse:
            mock_fileresponse.return_value = "html_response_object"
            response = client.get("/diagrams")
            assert response.status_code == 200

    def test_get_diagrams_html_page_not_found(self):
        """GET /diagrams returns 404 if HTML file does not exist."""
        with patch('os.path.exists', return_value=False):
            response = client.get("/diagrams")
            assert response.status_code == 404

    def test_get_mermaid_png_path_cached(self):
        """get_mermaid_png_path returns cached file if exists."""
        from backend.main import get_mermaid_png_path
        def exists_side_effect(path):
            if str(path).endswith('.png'):
                return True
            return False
        with patch('os.path.exists', side_effect=exists_side_effect):
            res = get_mermaid_png_path("sys_arch")
            assert res.endswith(".png")

    def test_get_mermaid_png_path_fetch_success(self):
        """get_mermaid_png_path fetches from mermaid.ink if not cached."""
        from backend.main import get_mermaid_png_path
        from unittest.mock import mock_open
        def exists_side_effect(path):
            if str(path).endswith('.png'):
                return False
            return True
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = b"png_data"
        with patch('os.path.exists', side_effect=exists_side_effect), \
             patch('urllib.request.urlopen', return_value=mock_response), \
             patch('builtins.open', mock_open()), \
             patch('os.makedirs'):
            res = get_mermaid_png_path("sys_arch")
            assert res.endswith(".png")

    def test_get_mermaid_png_path_invalid_name(self):
        """get_mermaid_png_path raises ValueError for unknown diagram name."""
        from backend.main import get_mermaid_png_path
        with pytest.raises(ValueError):
            get_mermaid_png_path("invalid_diagram_name")

    def test_websocket_realtime_predict_no_data(self):
        """Websocket endpoint returns error if dataset doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with client.websocket_connect("/ws/realtime-predict") as websocket:
                data = websocket.receive_json()
                assert "error" in data
                assert "dataset not found" in data.get("error", "").lower()

    def test_websocket_realtime_predict_success(self):
        """Websocket endpoint streams predictions accurately."""
        import pandas as pd
        mock_sup_df = pd.DataFrame({
            'CustomerID': ['123', '456'],
            'Target': [1, 0],
            'Recency': [10.0, 20.0],
            'Frequency': [5.0, 10.0],
            'Monetary': [200.0, 500.0],
            'AvgOrderValue': [40.0, 50.0],
            'UniqueProducts': [3.0, 4.0],
            'ReturnRate': [0.0, 0.05],
            'CustomerLifetimeDays': [100.0, 150.0],
            'PurchaseFrequencyMonthly': [1.5, 2.0],
            'AvgQuantityPerOrder': [10.0, 12.0],
            'Recency_log': [1.0, 2.0],
            'Frequency_log': [1.0, 2.0],
            'Monetary_log': [1.0, 2.0],
        })
        
        mock_train_split = (None, mock_sup_df, None, mock_sup_df['Target'].values)
        
        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=mock_sup_df), \
             patch('backend.main.train_test_split', return_value=mock_train_split), \
             patch('asyncio.sleep', return_value=None):
            with client.websocket_connect("/ws/realtime-predict") as websocket:
                # First row
                data1 = websocket.receive_json()
                assert "customer_id" in data1
                assert data1["customer_id"] == "123"
                assert "running_accuracy" in data1
                
                # Second row
                data2 = websocket.receive_json()
                assert data2["customer_id"] == "456"

    def test_chat_endpoint_with_api_key(self):
        """Chat endpoint should call Groq API when API key is present."""
        import json
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "This is a mock reply from Groq AI"
                }
            }]
        }).encode("utf-8")
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "mock_groq_key"}), \
             patch('urllib.request.urlopen', return_value=mock_response):
            payload = {
                "message": "Specify Champions segment size?",
                "history": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
            }
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["reply"] == "This is a mock reply from Groq AI"

    def test_chat_endpoint_api_key_failure(self):
        """Chat endpoint handles urllib failures correctly."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "mock_groq_key"}), \
             patch('urllib.request.urlopen', side_effect=Exception("API Error")):
            payload = {
                "message": "Specify Champions segment size?",
                "history": []
            }
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Failed to get response" in data["error"]


class TestRetrainingPipeline:
    """Test suite for physical/simulated execute_pipeline_retraining execution."""

    def test_execute_pipeline_retraining_success(self):
        """execute_pipeline_retraining runs all phases and updates status on success."""
        from backend.main import execute_pipeline_retraining, retraining_status
        
        with patch('backend.pipeline.preprocessing.run_preprocessing'), \
             patch('backend.pipeline.rfm_features.run_rfm_engineering'), \
             patch('backend.pipeline.segmentation.run_segmentation'), \
             patch('backend.pipeline.model_training.run_model_training'), \
             patch('backend.pipeline.drift_detection.run_drift_detection'), \
             patch('backend.main.load_models_and_scaler'), \
             patch('backend.main.precompute_cached_data'):
            execute_pipeline_retraining()
            assert retraining_status["status"] == "success"
            assert retraining_status["progress"] == 100
            assert retraining_status["error"] is None

    def test_execute_pipeline_retraining_failure(self):
        """execute_pipeline_retraining updates status to failed when an exception occurs."""
        from backend.main import execute_pipeline_retraining, retraining_status
        
        with patch('backend.pipeline.preprocessing.run_preprocessing', side_effect=Exception("Preprocessing failed")):
            execute_pipeline_retraining()
            assert retraining_status["status"] == "failed"
            assert "Preprocessing failed" in retraining_status["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

