import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from yaml file"""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to CSV"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    get_logger(__name__).info(f"✅ Saved dataframe to {path}")

def load_dataframe(path: str) -> pd.DataFrame:
    """Load dataframe from CSV"""
    df = pd.read_csv(path)
    get_logger(__name__).info(f"✅ Loaded dataframe from {path} - Shape: {df.shape}")
    return df