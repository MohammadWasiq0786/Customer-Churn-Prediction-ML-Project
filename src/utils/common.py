import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict

def save_json(path: str, data: Dict):
    """Save data to JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> Dict:
    """Load data from JSON file"""
    with open(path, "r") as f:
        return json.load(f)

def save_pickle(path: str, obj: Any):
    """Save object to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """Load object from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)

def create_directories(paths: list):
    """Create multiple directories"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
