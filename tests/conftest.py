"""
Pytest configuration and fixtures
"""
import pytest
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@pytest.fixture(scope="session", autouse=True)
def ensure_models_exist():
    """Ensure model files exist before running tests"""
    model_path = Path("models/xgboost_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Check if real models exist
    models_exist = model_path.exists() and scaler_path.exists()
    
    if not models_exist:
        print("\n📦 Creating test models (real models not found)...")
        
        # Create dummy model for testing
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 30)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create dummy scaler
        scaler = StandardScaler()
        scaler.fit(X_dummy)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✅ Test models created successfully")
    else:
        print("\n✅ Using existing trained models for testing")

@pytest.fixture(scope="module")
def test_client():
    """Create a test client with properly loaded models"""
    from fastapi.testclient import TestClient
    from app.main import app, load_model
    
    # Force load the model
    load_model()
    
    with TestClient(app) as client:
        yield client