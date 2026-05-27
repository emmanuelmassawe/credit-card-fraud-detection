"""
Model Tests for Credit Card Fraud Detection
"""
import pytest
import pickle
import numpy as np
from pathlib import Path

def test_model_exists():
    """Test if model file exists"""
    model_path = Path("models/xgboost_model.pkl")
    assert model_path.exists(), "Model file not found"

def test_scaler_exists():
    """Test if scaler file exists"""
    scaler_path = Path("models/scaler.pkl")
    assert scaler_path.exists(), "Scaler file not found"

def test_model_loads():
    """Test if model can be loaded"""
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None

def test_scaler_loads():
    """Test if scaler can be loaded"""
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert scaler is not None

def test_model_prediction():
    """Test if model can make predictions"""
    # Load model and scaler
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Create sample data
    sample_data = np.random.randn(1, 30)
    
    # Scale
    scaled_data = scaler.transform(sample_data)
    
    # Predict
    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    
    assert prediction is not None
    assert probabilities is not None
    assert prediction[0] in [0, 1]
    assert len(probabilities[0]) == 2
    assert 0 <= probabilities[0][0] <= 1
    assert 0 <= probabilities[0][1] <= 1