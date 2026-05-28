"""
API Tests for Credit Card Fraud Detection
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app  # noqa: E402


def test_read_root():
    """Test API root endpoint"""
    with TestClient(app) as client:
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "Credit Card Fraud Detection API"
        assert "version" in data
        print(f"\n✅ Root: {data}")


def test_health_check():
    """Test health endpoint"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        print(f"\n✅ Health: {data}")


def test_predict_normal_transaction():
    """Test prediction with normal transaction"""
    transaction = {
        "Time": 0.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=transaction)

        if response.status_code != 200:
            print(f"\n❌ Status: {response.status_code}")
            print(f"Response: {response.json()}")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"

        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["Normal", "Fraud"]
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 100.0

        print(f"\n✅ Normal Transaction Prediction: {data['prediction_label']}")
        print(f"   Fraud Prob: {data['fraud_probability']:.4f}")
        print(f"   Confidence: {data['confidence']:.2f}%")


def test_predict_fraud_transaction():
    """Test prediction with fraud transaction"""
    transaction = {
        "Time": 406.0,
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724120274487,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076182388,
        "V24": 0.320198198514526,
        "V25": 0.0445191674731724,
        "V26": 0.177839798284401,
        "V27": 0.261145002567677,
        "V28": -0.143275874698919,
        "Amount": 0.0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=transaction)

        if response.status_code != 200:
            print(f"\n❌ Status: {response.status_code}")
            print(f"Response: {response.json()}")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"

        data = response.json()
        assert "prediction" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 100.0

        print(f"\n✅ Fraud Transaction Prediction: {data['prediction_label']}")
        print(f"   Fraud Prob: {data['fraud_probability']:.4f}")
        print(f"   Confidence: {data['confidence']:.2f}%")


def test_predict_invalid_data():
    """Test prediction with missing fields"""
    invalid_transaction = {
        "Time": 0,
        "Amount": 100,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422


def test_predict_invalid_types():
    """Test prediction with invalid data types"""
    invalid_transaction = {
        "Time": "invalid",
        "V1": 0,
        "V2": 0,
        "V3": 0,
        "V4": 0,
        "V5": 0,
        "V6": 0,
        "V7": 0,
        "V8": 0,
        "V9": 0,
        "V10": 0,
        "V11": 0,
        "V12": 0,
        "V13": 0,
        "V14": 0,
        "V15": 0,
        "V16": 0,
        "V17": 0,
        "V18": 0,
        "V19": 0,
        "V20": 0,
        "V21": 0,
        "V22": 0,
        "V23": 0,
        "V24": 0,
        "V25": 0,
        "V26": 0,
        "V27": 0,
        "V28": 0,
        "Amount": 100,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422
