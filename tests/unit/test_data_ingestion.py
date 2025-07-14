import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from src.data.data_ingestion import DataIngestion
from src.config.configuration import DataIngestionConfig

class TestDataIngestion:
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        config = DataIngestionConfig(
            raw_data_path="test_data.csv",
            processed_data_path="processed/",
            test_size=0.2,
            random_state=42
        )
        
        data_ingestion = DataIngestion(config)
        df = data_ingestion.generate_sample_data()
        
        # Check data shape and columns
        assert df.shape[0] == 10000
        assert 'churn' in df.columns
        assert 'customer_id' in df.columns
        assert df['churn'].nunique() == 2  # Binary target
        
    def test_data_ingestion_pipeline(self, temp_dir):
        """Test complete data ingestion pipeline"""
        config = DataIngestionConfig(
            raw_data_path=f"{temp_dir}/raw_data.csv",
            processed_data_path=f"{temp_dir}/processed/",
            test_size=0.2,
            random_state=42
        )
        
        data_ingestion = DataIngestion(config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Check if files are created
        assert Path(train_path).exists()
        assert Path(test_path).exists()
        
        # Check data split
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        total_samples = len(train_df) + len(test_df)
        assert abs(len(test_df) / total_samples - 0.2) < 0.01  # Approximately 20% test
