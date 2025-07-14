import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger
from src.config.configuration import DataIngestionConfig
from pathlib import Path

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample customer data for demonstration"""
        np.random.seed(self.config.random_state)
        n_samples = 10000
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 15, n_samples),
            'tenure': np.random.exponential(2, n_samples),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(1500, 800, n_samples),
            'contract_length': np.random.choice([1, 12, 24], n_samples, p=[0.3, 0.4, 0.3]),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples, p=[0.4, 0.5, 0.1]),
            'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        }
        
        # Create churn based on business logic
        df = pd.DataFrame(data)
        churn_probability = (
            0.1 +  # Base churn rate
            0.3 * (df['monthly_charges'] > 80) +  # High monthly charges
            0.2 * (df['tenure'] < 1) +  # Low tenure
            0.15 * (df['contract_length'] == 1) +  # Month-to-month contract
            0.1 * (df['tech_support'] == 'No')  # No tech support
        )
        
        df['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_samples)
        
        return df
    
    def initiate_data_ingestion(self) -> tuple:
        """Main method to handle data ingestion"""
        try:
            logger.info("Starting data ingestion")
            
            # Create directories
            Path(self.config.raw_data_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.processed_data_path).mkdir(parents=True, exist_ok=True)
            
            # Generate or load data
            if not Path(self.config.raw_data_path).exists():
                logger.info("Generating sample data")
                df = self.generate_sample_data()
                df.to_csv(self.config.raw_data_path, index=False)
            else:
                logger.info("Loading existing data")
                df = pd.read_csv(self.config.raw_data_path)
            
            logger.info(f"Data shape: {df.shape}")
            
            # Split data
            train_df, test_df = train_test_split(
                df, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=df['churn']
            )
            
            # Save splits
            train_path = Path(self.config.processed_data_path) / "train.csv"
            test_path = Path(self.config.processed_data_path) / "test.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info("Data ingestion completed successfully")
            return str(train_path), str(test_path)
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise e
