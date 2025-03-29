import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def ridge(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    copy_X: bool = True,
    max_iter: int = None,
    tol: float = 0.0001,
    solver: str = 'auto',
    positive: bool = False,
    random_state=None
):
    """
    Trains a Ridge Regression model using the provided data and hyperparameters.

    Hyperparameters:
      - alpha: float (default 1.0)
      - fit_intercept: bool (default True)
      - copy_X: bool (default True)
      - max_iter: int or None (default None)
      - tol: float (default 0.0001)
      - solver: str, {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'} (default 'auto')
      - positive: bool (default False)
      - random_state: int or None (default None)

    For regression, evaluation metrics include Mean Squared Error (MSE) and R² score.
    
    After training, the model is serialized using pickle along with a record of the training
    progress (currently, a single value representing the final training MSE). The complete artifact
    package, which includes a generated _id, model name, training time, metrics, and the pickled model,
    is stored in the dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the Ridge model with provided hyperparameters.
        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state
        )

        logging.info("Starting training for Ridge Regression.")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate predictions on training and test datasets.
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Record a simple progress metric (here, we record the final training MSE).
        progress = [train_mse]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "Ridge",
            "train_time": train_time,
            "metrics": {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2,
            },
            "progress": progress,
            "trained_model": model  # This will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"Ridge Regression trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "Ridge",
            "train_time": train_time,
            "metrics": {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2,
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object as well.
        }

    except Exception as e:
        logging.error(f"Error in Ridge training: {e}")
        raise CustomException(e, sys)
