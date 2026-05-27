import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import get_logger, load_params, save_dataframe

logger = get_logger(__name__)

def ingest_data(params: dict) -> str:
    """Load Credit Card Fraud Detection dataset"""
    logger.info("🚀 Starting Data Ingestion...")
    
    raw_data_path = params["data"]["raw_data_path"]
    
    # Check if file exists
    if not Path(raw_data_path).exists():
        logger.error(f"❌ File not found: {raw_data_path}")
        logger.error("Please download the dataset from Kaggle first!")
        raise FileNotFoundError(f"Dataset not found at {raw_data_path}")
    
    # Load dataset
    logger.info(f"📂 Loading dataset from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Display dataset info
    logger.info(f"📊 Dataset shape: {df.shape}")
    logger.info(f"📊 Columns: {list(df.columns)}")
    logger.info(f"📊 Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Check target distribution
    target_col = params["base"]["target_column"]
    logger.info(f"📊 Target column: {target_col}")
    logger.info(f"📊 Class distribution:\n{df[target_col].value_counts()}")
    logger.info(f"📊 Class percentages:\n{df[target_col].value_counts(normalize=True) * 100}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    logger.info(f"📊 Missing values: {missing}")
    
    logger.info(f"✅ Data Ingestion complete!")
    
    return raw_data_path

if __name__ == "__main__":
    params = load_params()
    ingest_data(params)