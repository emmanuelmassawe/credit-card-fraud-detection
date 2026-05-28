from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import logging
import pickle
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== LOAD MODEL ====================
model = None
scaler = None


def load_model():
    """Load model and scaler from disk"""
    global model, scaler
    try:
        with open("models/xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded successfully")

        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info("✅ Scaler loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler"""
    logger.info("🚀 Starting up...")
    load_model()
    logger.info("✅ Application ready!")
    yield
    logger.info("🛑 Shutting down...")


# ==================== CREATE APP ====================
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection using XGBoost + SMOTE",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== SCHEMAS ====================
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    fraud_probability: float
    confidence: float


# ==================== SERVE FRONTEND ====================
# CORRECT - mounting frontend/static as /static
if Path("frontend/static").exists():
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Also need to mount root for index.html
if Path("frontend").exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse("frontend/index.html")


# ==================== API ENDPOINTS ====================
@app.get("/api")
async def api_root():
    return {
        "app": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None and scaler is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: TransactionInput):
    """Predict if a credit card transaction is fraudulent"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler not loaded.",
        )

    try:
        data = transaction.dict()
        df = pd.DataFrame([data])

        logger.info(f"📥 Transaction: Amount=${data['Amount']:.2f}")

        scaled_data = scaler.transform(df)
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        fraud_probability = float(probabilities[1])
        confidence = (
            fraud_probability * 100
            if prediction == 1
            else (1 - fraud_probability) * 100
        )

        result = {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Normal",
            "fraud_probability": round(fraud_probability, 4),
            "confidence": round(confidence, 2),
        }

        logger.info(f"✅ Result: {result['prediction_label']} | Prob: {fraud_probability:.4f}")
        return result

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)