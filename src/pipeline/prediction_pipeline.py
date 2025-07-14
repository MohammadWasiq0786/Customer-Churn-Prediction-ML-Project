import joblib
import pandas as pd
from src.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/model.pkl", 
                 preprocessor_path: str = "artifacts/preprocessor.pkl"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessor"""
        try:
            if Path(self.model_path).exists() and Path(self.preprocessor_path).exists():
                self.model = joblib.load(self.model_path)
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("Model and preprocessor loaded successfully")
            else:
                logger.warning("Model artifacts not found. Please train the model first.")
                raise FileNotFoundError("Model artifacts not found")
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise e
    
    def predict(self, features: pd.DataFrame):
        """Make predictions on new data"""
        try:
            # Transform features
            features_transformed = self.preprocessor.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_transformed)
            probabilities = self.model.predict_proba(features_transformed)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise e
    
    def predict_single(self, features_dict: dict):
        """Make prediction for a single instance"""
        try:
            df = pd.DataFrame([features_dict])
            predictions, probabilities = self.predict(df)
            return int(predictions[0]), float(probabilities[0])
            
        except Exception as e:
            logger.error(f"Error in single prediction: {str(e)}")
            raise e
