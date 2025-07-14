import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.pipeline.prediction_pipeline import PredictionPipeline

class TestPredictionPipeline:
    
    @patch('joblib.load')
    @patch('pathlib.Path.exists')
    def test_prediction_pipeline_initialization(self, mock_exists, mock_load):
        """Test prediction pipeline initialization"""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_load.side_effect = [mock_model, mock_preprocessor]
        
        pipeline = PredictionPipeline()
        
        assert pipeline.model is not None
        assert pipeline.preprocessor is not None
        
    def test_predict_single(self):
        """Test single prediction"""
        # Mock the pipeline components
        pipeline = PredictionPipeline.__new__(PredictionPipeline)
        pipeline.model = Mock()
        pipeline.preprocessor = Mock()
        
        # Setup mock returns
        pipeline.preprocessor.transform.return_value = np.array([[1, 2, 3]])
        pipeline.model.predict.return_value = np.array([1])
        pipeline.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        features_dict = {
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
        
        prediction, probability = pipeline.predict_single(features_dict)
        
        assert prediction == 1
        assert probability == 0.7
