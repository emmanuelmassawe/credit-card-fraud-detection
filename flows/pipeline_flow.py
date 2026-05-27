import os
import sys
from prefect import flow, task, get_run_logger
from prefect.task_runners import ThreadPoolTaskRunner
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_params
from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

# ============================================================
# PREFECT TASKS
# ============================================================

@task(name="Data Ingestion", retries=2, retry_delay_seconds=30)
def data_ingestion_task(params: dict) -> str:
    """Task: Ingest credit card fraud dataset"""
    logger = get_run_logger()
    logger.info("📥 Running Data Ingestion Task...")
    result = ingest_data(params)
    logger.info(f"✅ Data validated: {result}")
    return result

@task(name="Data Preprocessing & SMOTE", retries=2, retry_delay_seconds=30)
def preprocessing_task(params: dict) -> tuple:
    """Task: Preprocess data and apply SMOTE sampling"""
    logger = get_run_logger()
    logger.info("⚙️ Running Preprocessing & SMOTE Task...")
    train_path, test_path = preprocess_data(params)
    logger.info(f"✅ Train: {train_path} | Test: {test_path}")
    return train_path, test_path

@task(name="XGBoost Training", retries=1, retry_delay_seconds=60)
def training_task(params: dict) -> str:
    """Task: Train XGBoost model"""
    logger = get_run_logger()
    logger.info("🏋️ Running XGBoost Training Task...")
    model_path = train_model(params)
    logger.info(f"✅ Model saved to: {model_path}")
    return model_path

@task(name="Model Evaluation")
def evaluation_task(params: dict) -> dict:
    """Task: Evaluate model and generate metrics"""
    logger = get_run_logger()
    logger.info("📊 Running Model Evaluation Task...")
    metrics = evaluate_model(params)
    logger.info(f"✅ Metrics: {metrics}")
    return metrics

@task(name="DVC Push to DagsHub")
def dvc_push_task() -> None:
    """Task: Push data/models to DagsHub with DVC"""
    logger = get_run_logger()
    logger.info("📤 Pushing artifacts to DagsHub with DVC...")
    os.system("dvc add data/processed models reports")
    os.system("dvc push")
    logger.info("✅ DVC Push complete!")

@task(name="Git Commit & Push")
def git_push_task(metrics: dict) -> None:
    """Task: Commit results to Git"""
    logger = get_run_logger()
    logger.info("📤 Committing results to Git...")

    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_score", 0)
    roc_auc = metrics.get("roc_auc", 0)

    os.system("git add .")
    os.system(f'git commit -m "Pipeline run: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}"')
    os.system("git push origin main")

    logger.info("✅ Git Push complete!")

# ============================================================
# MAIN FLOW
# ============================================================

@flow(
    name="Credit Card Fraud Detection Pipeline",
    description="End-to-end MLOps pipeline: XGBoost + SMOTE + MLflow + DVC",
    task_runner = ThreadPoolTaskRunner(),
    version="1.0.0"
)
def fraud_detection_pipeline(config_path: str = "params.yaml"):
    """
    Complete MLOps Pipeline:
    1. Data Ingestion      (validate Kaggle dataset)
    2. Preprocessing       (scaling + SMOTE)
    3. XGBoost Training    (MLflow tracking)
    4. Evaluation          (metrics + plots)
    5. DVC Push            (version data/models)
    6. Git Push            (version code)
    """
    logger = get_run_logger()
    logger.info("=" * 70)
    logger.info("🚀 Starting Credit Card Fraud Detection Pipeline")
    logger.info("=" * 70)

    # Load parameters
    params = load_params(config_path)

    # ─── Step 1 ───────────────────────────────────────────
    logger.info("📍 Step 1/6: Data Ingestion")
    raw_data_path = data_ingestion_task(params)

    # ─── Step 2 ───────────────────────────────────────────
    logger.info("📍 Step 2/6: Data Preprocessing & SMOTE")
    train_path, test_path = preprocessing_task(params)

    # ─── Step 3 ───────────────────────────────────────────
    logger.info("📍 Step 3/6: XGBoost Training")
    model_path = training_task(params)

    # ─── Step 4 ───────────────────────────────────────────
    logger.info("📍 Step 4/6: Model Evaluation")
    metrics = evaluation_task(params)

    # ─── Step 5 ───────────────────────────────────────────
    logger.info("📍 Step 5/6: DVC Push to DagsHub")
    dvc_push_task()

    # ─── Step 6 ───────────────────────────────────────────
    logger.info("📍 Step 6/6: Git Commit & Push")
    git_push_task(metrics)

    # ─── Summary ──────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("🎉 Pipeline Completed Successfully!")
    logger.info("=" * 70)
    logger.info("📊 Final Metrics:")
    logger.info(f"   • Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"   • Precision: {metrics['precision']:.4f}")
    logger.info(f"   • Recall:    {metrics['recall']:.4f}")
    logger.info(f"   • F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"   • ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("=" * 70)

    return metrics

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    fraud_detection_pipeline()