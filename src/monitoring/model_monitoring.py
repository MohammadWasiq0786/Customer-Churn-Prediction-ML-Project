import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from src.utils.logger import setup_logger
from src.monitoring.data_drift import DataDriftDetector
import json

logger = setup_logger(__name__)

class ModelMonitor:
    def __init__(self, model_path: str, preprocessor_path: str, reference_data: pd.DataFrame):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.reference_data = reference_data
        self.drift_detector = DataDriftDetector(reference_data)
        self.monitoring_data = []
    
    def monitor_prediction_quality(self, predictions: np.array, probabilities: np.array):
        """Monitor prediction quality metrics"""
        try:
            metrics = {
                'avg_probability': float(np.mean(probabilities)),
                'prediction_distribution': {
                    'class_0': int(np.sum(predictions == 0)),
                    'class_1': int(np.sum(predictions == 1))
                },
                'high_confidence_predictions': int(np.sum((probabilities > 0.8) | (probabilities < 0.2))),
                'low_confidence_predictions': int(np.sum((probabilities >= 0.4) & (probabilities <= 0.6)))
            }
            
            logger.info(f"Prediction quality metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in prediction quality monitoring: {str(e)}")
            return {}
    
    def monitor_data_drift(self, current_data: pd.DataFrame):
        """Monitor for data drift"""
        try:
            drift_report = self.drift_detector.detect_drift(current_data)
            
            # Count drifted features
            drifted_features = [col for col, report in drift_report.items() if report['drift_detected']]
            
            monitoring_result = {
                'timestamp': datetime.now().isoformat(),
                'total_features': len(drift_report),
                'drifted_features_count': len(drifted_features),
                'drifted_features': drifted_features,
                'drift_report': drift_report
            }
            
            logger.info(f"Data drift monitoring: {len(drifted_features)} out of {len(drift_report)} features drifted")
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Error in data drift monitoring: {str(e)}")
            return {}
    
    def log_monitoring_data(self, monitoring_data: dict, log_file: str = "logs/monitoring.json"):
        """Log monitoring data to file"""
        try:
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            existing_data = []
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            
            # Append new data
            existing_data.append(monitoring_data)
            
            # Keep only last 1000 records
            existing_data = existing_data[-1000:]
            
            # Save updated data
            with open(log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging monitoring data: {str(e)}")
