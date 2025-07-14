import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.preprocessor = None
        self.label_encoders = {}
    
    def get_data_transformer(self, df: pd.DataFrame):
        """Create preprocessing pipeline"""
        try:
            # Identify column types
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'churn' in numeric_features:
                numeric_features.remove('churn')
            if 'customer_id' in numeric_features:
                numeric_features.remove('customer_id')
                
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Numeric features: {numeric_features}")
            logger.info(f"Categorical features: {categorical_features}")
            
            # Create preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating data transformer: {str(e)}")
            raise e
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Transform training and test data"""
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Data transformation started")
            
            # Drop customer_id
            if 'customer_id' in train_df.columns:
                train_df = train_df.drop('customer_id', axis=1)
                test_df = test_df.drop('customer_id', axis=1)
            
            # Separate features and target
            target_column = 'churn'
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Get and fit preprocessor
            preprocessing_obj = self.get_data_transformer(X_train)
            
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            # Save preprocessor
            preprocessor_path = Path("artifacts/preprocessor.pkl")
            preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessing_obj, preprocessor_path)
            
            logger.info("Data transformation completed successfully")
            
            return (
                X_train_transformed, y_train,
                X_test_transformed, y_test,
                str(preprocessor_path)
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise e
