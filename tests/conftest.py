"""
Pytest configuration and fixtures
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session", autouse=True)
def ensure_models_exist():
    """Ensure model files exist before running tests"""
    model_path = Path("models/xgboost_model.pkl")
    scaler_path = Path("models/scaler.pkl")

    Path("models").mkdir(exist_ok=True)

    if not model_path.exists() or not scaler_path.exists():
        print("\n📦 Creating test models...")

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 30)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("✅ Test model created")

        scaler = StandardScaler()
        scaler.fit(X_dummy)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print("✅ Test scaler created")
    else:
        print("\n✅ Using existing trained models")
