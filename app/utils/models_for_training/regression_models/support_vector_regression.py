import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def svr(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    kernel: str = 'rbf',
    degree: int = 3,
    gamma: str = 'scale',
    coef0: float = 0.0,
    tol: float = 0.001,
    C: float = 1.0,
    epsilon: float = 0.1,
    shrinking: bool = True,
    cache_size: float = 200,
    verbose: bool = False,
    max_iter: int = -1
):
    """
    Trains an SVR (Support Vector Regressor) model using the provided data and hyperparameters.
    
    Hyperparameters:
      - kernel: str, e.g., 'rbf', 'linear', 'poly', 'sigmoid' (default 'rbf')
      - degree: int (default 3, used if kernel is 'poly')
      - gamma: {'scale', 'auto'} or float (default 'scale')
      - coef0: float (default 0.0)
      - tol: float (default 0.001)
      - C: float (default 1.0)
      - epsilon: float (default 0.1)
      - shrinking: bool (default True)
      - cache_size: float (default 200)
      - verbose: bool (default False)
      - max_iter: int (default -1, i.e., no limit)
      
    For regression, evaluation metrics include Mean Squared Error (MSE) and R² score.
    
    After training, the model is serialized with pickle along with a record of the training progress
    (here, a single value representing the final training MSE). The complete artifact package, which
    includes a generated _id, model name, training time, metrics, and the pickled model, is stored in the
    dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the SVR model with the provided hyperparameters.
        model = SVR(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter
        )

        logging.info("Starting training for SVR.")
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

        # Record a simple progress metric (here, the final training MSE).
        progress = [train_mse]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "SVR",
            "train_time": train_time,
            "metrics": {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2
            },
            "progress": progress,
            "trained_model": model  # This will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"SVR trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "SVR",
            "train_time": train_time,
            "metrics": {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object as well.
        }

    except Exception as e:
        logging.error(f"Error in SVR training: {e}")
        raise CustomException(e, sys)
