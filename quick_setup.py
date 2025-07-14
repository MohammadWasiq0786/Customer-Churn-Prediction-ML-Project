#!/usr/bin/env python3
"""
Quick setup script to create all missing files
"""

import os
import yaml
from pathlib import Path

def create_config():
    """Create the config.yaml file"""
    
    config = {
        'data': {
            'raw_data_path': 'data/raw/customer_data.csv',
            'processed_data_path': 'data/processed/',
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'model_name': 'RandomForestClassifier',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'target_column': 'churn'
        },
        'training': {
            'experiment_name': 'churn_prediction',
            'registered_model_name': 'churn_model'
        },
        'api': {
            'host': '127.0.0.1',
            'port': 8000
        },
        'monitoring': {
            'drift_threshold': 0.05,
            'performance_threshold': 0.85
        }
    }
    
    # Create directories
    Path('config').mkdir(exist_ok=True)
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Write config file
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created config/config.yaml")

def create_schemas_if_missing():
    """Create schemas.py if it doesn't exist"""
    
    schemas_path = Path('api/schemas.py')
    if not schemas_path.exists():
        schemas_content = '''from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Customer age")
    tenure: float = Field(..., ge=0, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_length: int = Field(..., ge=1, le=24, description="Contract length in months")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Online security service")
    tech_support: str = Field(..., description="Tech support service")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int = Field(..., description="0 for no churn, 1 for churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_version: str
    uptime_seconds: Optional[float] = None
'''
        
        with open(schemas_path, 'w') as f:
            f.write(schemas_content)
        
        print("✅ Created api/schemas.py")

def main():
    """Main setup function"""
    print("🔧 Setting up missing files...")
    
    create_config()
    create_schemas_if_missing()
    
    print("\n🎉 Setup complete!")
    print("Now try running: python api/app.py")

if __name__ == "__main__":
    main()