import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(40, 15, n_samples),
        'tenure': np.random.exponential(2, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(1500, 800, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber Optic'], n_samples),
        'online_security': np.random.choice(['Yes', 'No'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_features():
    """Sample customer features for API testing"""
    return {
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
