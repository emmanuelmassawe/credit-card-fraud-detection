import pickle
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils import get_logger, load_dataframe, load_params, save_dataframe

logger = get_logger(__name__)

SAMPLERS = {"SMOTE": SMOTE, "ADASYN": ADASYN, "RandomOverSampler": RandomOverSampler, "SMOTETomek": SMOTETomek}

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def apply_sampling(X_train, y_train, params: dict):
    """Apply imbalanced sampling strategy"""

    sampling_strategy = params["sampling"]["strategy"]
    logger.info(f"🔄 Applying {sampling_strategy} sampling...")

    sampler_class = SAMPLERS.get(sampling_strategy, SMOTE)

    sampler_kwargs = {
        "random_state": params["base"]["random_state"],
        "sampling_strategy": params["sampling"]["sampling_strategy"],
    }

    # SMOTE specific params
    if sampling_strategy in ["SMOTE", "SMOTETomek"]:
        sampler_kwargs["k_neighbors"] = params["sampling"]["k_neighbors"]

    sampler = sampler_class(**sampler_kwargs)

    logger.info(f"📊 Before sampling: {pd.Series(y_train).value_counts().to_dict()}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    logger.info(f"📊 After sampling:  {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled


def preprocess_data(params: dict) -> tuple:
    """Main preprocessing function"""
    logger.info("🚀 Starting Data Preprocessing...")

    # Load raw data
    df = load_dataframe(params["data"]["raw_data_path"])
    target_col = params["base"]["target_column"]

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"📊 Features shape: {X.shape}")
    logger.info(f"📊 Feature columns: {list(X.columns)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["base"]["test_size"], random_state=params["base"]["random_state"], stratify=y
    )

    logger.info(f"📊 Train set: {X_train.shape}")
    logger.info(f"📊 Test set: {X_test.shape}")

    # Apply scaling
    scaler_name = params["preprocessing"]["scaler"]
    logger.info(f"🔄 Applying {scaler_name} scaling...")

    scaler = SCALERS[scaler_name]()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    Path("models").mkdir(exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("✅ Scaler saved to models/scaler.pkl")

    # Apply SMOTE ONLY on training data
    X_train_resampled, y_train_resampled = apply_sampling(X_train_scaled, y_train.values, params)

    # Convert back to DataFrames
    train_df = pd.DataFrame(X_train_resampled, columns=X.columns)
    train_df[target_col] = y_train_resampled

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df[target_col] = y_test.values

    # Save processed data
    save_dataframe(train_df, params["data"]["train_path"])
    save_dataframe(test_df, params["data"]["test_path"])

    logger.info("✅ Data Preprocessing complete!")
    return params["data"]["train_path"], params["data"]["test_path"]


if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)
