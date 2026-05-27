import pandas as pd
import numpy as np
import pickle
import json
import mlflow
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from dotenv import load_dotenv
from src.utils import get_logger, load_params, load_dataframe

load_dotenv()
logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"])
    plt.title("Confusion Matrix - Credit Card Fraud Detection")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"📊 Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_prob, save_path: str):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"📊 ROC curve saved to {save_path}")

def plot_precision_recall_curve(y_true, y_prob, save_path: str):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Fraud Detection")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"📊 PR curve saved to {save_path}")

def evaluate_model(params: dict) -> dict:
    """Main evaluation function"""
    logger.info("🚀 Starting Model Evaluation...")
    
    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Load test data
    test_df = load_dataframe(params["data"]["test_path"])
    target_col = params["base"]["target_column"]
    
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    logger.info(f"📊 Test data shape: {X_test.shape}")
    logger.info(f"📊 Test labels distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Load model
    model_path = "models/xgboost_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"✅ Model loaded from {model_path}")
    
    # Load run_id
    with open("models/run_id.txt", "r") as f:
        run_id = f.read().strip()
    
    # Predictions
    logger.info("🔮 Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    logger.info("=" * 60)
    logger.info("📊 EVALUATION METRICS:")
    logger.info("=" * 60)
    logger.info(f"   ✅ Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"   ✅ Precision: {metrics['precision']:.4f}")
    logger.info(f"   ✅ Recall:    {metrics['recall']:.4f}")
    logger.info(f"   ✅ F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"   ✅ ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("=" * 60)
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Save metrics JSON
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("✅ Metrics saved to reports/metrics.json")
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"])
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)
    logger.info("✅ Classification report saved")
    
    print("\n" + report)
    
    # Generate plots
    logger.info("📊 Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, "reports/confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, "reports/roc_curve.png")
    plot_precision_recall_curve(y_test, y_prob, "reports/pr_curve.png")
    
    # Log to MLflow (resume existing run)
    with mlflow.start_run(run_id=run_id):
        logger.info("📤 Logging metrics to MLflow...")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/confusion_matrix.png")
        mlflow.log_artifact("reports/roc_curve.png")
        mlflow.log_artifact("reports/pr_curve.png")
        mlflow.log_artifact("reports/classification_report.txt")
    
    logger.info("✅ Evaluation complete!")
    logger.info(f"✅ Check MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}")
    
    return metrics

if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)