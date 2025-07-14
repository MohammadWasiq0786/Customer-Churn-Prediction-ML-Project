import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    processed_data_path: str
    test_size: float
    random_state: int

@dataclass
class ModelTrainingConfig:
    model_name: str
    hyperparameters: Dict[str, Any]
    target_column: str

@dataclass
class ApiConfig:
    host: str
    port: int

@dataclass
class MonitoringConfig:
    drift_threshold: float
    performance_threshold: float

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config/config.yaml"):
        self.config_filepath = config_filepath
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        with open(self.config_filepath, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data"]
        return DataIngestionConfig(**config)
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config["model"]
        return ModelTrainingConfig(**config)
    
    def get_api_config(self) -> ApiConfig:
        config = self.config["api"]
        return ApiConfig(**config)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        config = self.config["monitoring"]
        return MonitoringConfig(**config)
