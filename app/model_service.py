import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and using the XGBoost fraud detection model"""

    def __init__(self, model_path: str = "models/xgboost_model.pkl", scaler_path: str = "models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model
            if Path(self.model_path).exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Model loaded from {self.model_path}")
            else:
                logger.error(f"❌ Model not found at {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Load scaler
            if Path(self.scaler_path).exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                logger.error(f"❌ Scaler not found at {self.scaler_path}")
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

            # Define feature names (same as in dataset)
            self.feature_names = [
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount",
            ]

        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise

    def preprocess(self, data: dict) -> np.ndarray:
        """Preprocess input data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct feature order
        df = df[self.feature_names]

        # Scale features
        scaled_data = self.scaler.transform(df)

        return scaled_data

    def predict(self, data: dict) -> Tuple[int, float]:
        """
        Make prediction on single transaction

        Returns:
            prediction (int): 0 = Normal, 1 = Fraud
            probability (float): Probability of fraud
        """
        # Preprocess
        processed_data = self.preprocess(data)

        # Predict
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0]

        # Get fraud probability (class 1)
        fraud_prob = float(probability[1])

        return int(prediction), fraud_prob

    def batch_predict(self, data_list: list) -> list:
        """Make predictions on multiple transactions"""
        results = []
        for data in data_list:
            prediction, probability = self.predict(data)
            results.append((prediction, probability))
        return results

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.scaler is not None


# Global model service instance
model_service = None


def get_model_service() -> ModelService:
    """Get or create model service instance"""
    global model_service
    if model_service is None:
        model_service = ModelService()
    return model_service
