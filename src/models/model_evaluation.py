import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelEvaluation:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        try:
            # Transform test data
            X_test_transformed = self.preprocessor.transform(X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_transformed)
            y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred)
            
            logger.info("Model evaluation completed")
            
            return {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('artifacts/confusion_matrix.png')
        plt.close()
