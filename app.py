# ============================================================
# Wine Quality Prediction with MLflow + DagsHub
# ============================================================
# Dataset:
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez et al., Decision Support Systems, 2009
# ============================================================

import os
import sys
import warnings
import logging
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ============================================================
# Initialize DagsHub + MLflow (IMPORTANT)
# ============================================================
dagshub.init(
    repo_owner="OgRahuldutta",
    repo_name="ML_flow",
    mlflow=True
)

# ============================================================
# Utility function
# ============================================================
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Failed to download dataset. Check internet connection.\nError: %s",
            e,
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------
    train, test = train_test_split(
        data, test_size=0.25, random_state=42
    )

    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)

    # FIX: y must be 1D
    train_y = train["quality"]
    test_y = test["quality"]

    # --------------------------------------------------------
    # Hyperparameters (CLI optional)
    # --------------------------------------------------------
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # --------------------------------------------------------
    # MLflow Run
    # --------------------------------------------------------
    with mlflow.start_run():

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )

        model.fit(train_x, train_y)

        predictions = model.predict(test_x)

        rmse, mae, r2 = eval_metrics(test_y, predictions)

        print(f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        # ----------------------------------------------------
        # Log parameters & metrics
        # ----------------------------------------------------
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ----------------------------------------------------
        # Log model with signature
        # ----------------------------------------------------
        signature = infer_signature(train_x, model.predict(train_x))

        tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_uri_type != "file":
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="ElasticnetWineModel",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
            )
