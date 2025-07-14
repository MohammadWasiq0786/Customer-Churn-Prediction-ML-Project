import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.main import app

client = TestClient(app)

class TestAPI:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Churn Prediction API" in response.text
        
    @patch('api.main.prediction_pipeline')  
    def test_predict_endpoint(self, mock_pipeline):
        """Test prediction endpoint"""
        # Mock the prediction pipeline
        mock_pipeline.predict_single.return_value = (1, 0.75)
        
        sample_data = {
            "age": 35.0,
            "tenure": 12.0,
            "monthly_charges": 75.5,
            "total_charges": 1200.0,
            "contract_length": 12,
            "payment_method": "Credit Card",
            "internet_service": "Fiber Optic",
            "online_security": "Yes",
            "tech_support": "Yes"
        }
        
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        
    def test_predict_endpoint_validation_error(self):
        """Test prediction endpoint with invalid data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "monthly_charges": 75.5,
            # Missing required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
