import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from src.utils.logger import setup_logger
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        
    def detect_numerical_drift(self, current_data: pd.DataFrame, column: str):
        """Detect drift in numerical columns using KS test"""
        try:
            if column not in self.reference_data.columns or column not in current_data.columns:
                return False, 1.0
            
            ref_values = self.reference_data[column].dropna()
            curr_values = current_data[column].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                return False, 1.0
            
            statistic, p_value = ks_2samp(ref_values, curr_values)
            
            drift_detected = p_value < self.threshold
            return drift_detected, p_value
            
        except Exception as e:
            logger.error(f"Error detecting numerical drift for {column}: {str(e)}")
            return False, 1.0
    
    def detect_categorical_drift(self, current_data: pd.DataFrame, column: str):
        """Detect drift in categorical columns using chi-square test"""
        try:
            if column not in self.reference_data.columns or column not in current_data.columns:
                return False, 1.0
            
            ref_counts = self.reference_data[column].value_counts()
            curr_counts = current_data[column].value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            # Avoid zero frequencies
            ref_aligned = [max(1, x) for x in ref_aligned]
            curr_aligned = [max(1, x) for x in curr_aligned]
            
            # Perform chi-square test
            contingency_table = [ref_aligned, curr_aligned]
            try:
                statistic, p_value, _, _ = chi2_contingency(contingency_table)
            except ValueError:
                return False, 1.0
            
            drift_detected = p_value < self.threshold
            return drift_detected, p_value
            
        except Exception as e:
            logger.error(f"Error detecting categorical drift for {column}: {str(e)}")
            return False, 1.0
    
    def detect_drift(self, current_data: pd.DataFrame):
        """Detect drift across all columns"""
        drift_report = {}
        
        for column in self.reference_data.columns:
            if column == 'churn':  # Skip target column
                continue
                
            if self.reference_data[column].dtype in ['int64', 'float64']:
                drift_detected, p_value = self.detect_numerical_drift(current_data, column)
            else:
                drift_detected, p_value = self.detect_categorical_drift(current_data, column)
            
            drift_report[column] = {
                'drift_detected': drift_detected,
                'p_value': p_value
            }
        
        return drift_report
