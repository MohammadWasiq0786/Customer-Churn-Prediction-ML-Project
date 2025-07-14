import pandas as pd
from src.utils.logger import setup_logger
from typing import Dict, List

logger = setup_logger(__name__)

class DataValidation:
    def __init__(self):
        self.required_columns = [
            'age', 'tenure', 'monthly_charges', 'total_charges',
            'contract_length', 'payment_method', 'internet_service',
            'online_security', 'tech_support', 'churn'
        ]
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate if dataframe has required columns"""
        try:
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            logger.info("Schema validation passed")
            return True
        except Exception as e:
            logger.error(f"Error in schema validation: {str(e)}")
            return False
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return metrics"""
        try:
            quality_report = {
                'total_rows': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Check for anomalies
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col != 'churn':
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    quality_report[f'{col}_outliers'] = outliers
            
            logger.info("Data quality validation completed")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {str(e)}")
            return {}
