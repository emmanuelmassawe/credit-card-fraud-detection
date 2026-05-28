import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier

from src.utils import get_logger, load_dataframe, load_params

load_dotenv()
logger = get_logger(__name__)


def train_model(params: dict) -> str:
    """Train XGBoost model with MLflow tracking"""
    logger.info("🚀 Starting Model Training...")

    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("credit-card-fraud-detection")

    # Load training data
    train_df = load_dataframe(params["data"]["train_path"])
    target_col = params["base"]["target_column"]

    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values

    logger.info(f"📊 Training data shape: {X_train.shape}")
    logger.info(f"📊 Training labels distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # Create XGBoost model
    model = XGBClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        learning_rate=params["model"]["learning_rate"],
        min_child_weight=params["model"]["min_child_weight"],
        subsample=params["model"]["subsample"],
        colsample_bytree=params["model"]["colsample_bytree"],
        scale_pos_weight=params["model"]["scale_pos_weight"],
        random_state=params["base"]["random_state"],
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="xgboost_fraud_detection") as run:
        logger.info(f"📊 MLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params(
            {
                "model_name": "xgboost",
                "n_estimators": params["model"]["n_estimators"],
                "max_depth": params["model"]["max_depth"],
                "learning_rate": params["model"]["learning_rate"],
                "min_child_weight": params["model"]["min_child_weight"],
                "subsample": params["model"]["subsample"],
                "colsample_bytree": params["model"]["colsample_bytree"],
                "scale_pos_weight": params["model"]["scale_pos_weight"],
                "sampling_strategy": params["sampling"]["strategy"],
                "test_size": params["base"]["test_size"],
                "random_state": params["base"]["random_state"],
                "scaler": params["preprocessing"]["scaler"],
            }
        )

        # Train model
        logger.info("🏋️ Training XGBoost model...")
        model.fit(X_train, y_train)

        # Save model
        Path("models").mkdir(exist_ok=True)
        model_path = "models/xgboost_model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"✅ Model saved to {model_path}")

        # Log model to MLflow
        mlflow.xgboost.log_model(model, "model", registered_model_name="credit-card-fraud-xgboost")

        mlflow.log_artifact(model_path)

        # Save run_id for evaluation step
        with open("models/run_id.txt", "w") as f:
            f.write(run.info.run_id)

        logger.info("✅ Model training complete!")
        logger.info(f"✅ MLflow Run ID: {run.info.run_id}")

    return model_path


if __name__ == "__main__":
    params = load_params()
    train_model(params)
