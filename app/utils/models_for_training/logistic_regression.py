import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def logistic_regression(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    penalty: str = 'l2',
    dual: bool = False,
    tol: float = 0.0001,
    C: float = 1.0,
    fit_intercept: bool = True,
    intercept_scaling: float = 1,
    class_weight=None,
    random_state=None,
    solver: str = 'liblinear',
    max_iter: int = 100,
    multi_class: str = 'ovr',
    verbose: int = 0
):
    """
    Trains a LogisticRegression model using the provided data and hyperparameters.
    The hyperparameters are declared as function arguments so that the frontend can pass
    the desired values.

    Hyperparameters:
      - penalty: str, {'l1', 'l2', 'elasticnet', 'none'} (default 'l2')
      - dual: bool (default False)
      - tol: float (default 0.0001)
      - C: float (default 1.0)
      - fit_intercept: bool (default True)
      - intercept_scaling: float (default 1)
      - class_weight: dict or 'balanced' (default None)
      - random_state: int (default None)
      - solver: str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'} (default 'liblinear')
      - max_iter: int (default 100)
      - multi_class: str, {'ovr', 'multinomial'} (default 'ovr')
      - verbose: int (default 0)
      
    After training, the model is serialized using pickle along with a record of the training
    accuracy progress (currently, a single value representing the final training accuracy).
    
    The complete artifact, including a generated _id, model name, training time, metrics, and the
    pickled model, is then stored in the database under the `models` array of the dataset document.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the LogisticRegression model with given hyperparameters
        model = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose
        )

        logging.info("Starting training for LogisticRegression.")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate model predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        precision = precision_score(y_test, y_test_pred, average="weighted")
        recall = recall_score(y_test, y_test_pred, average="weighted")
        
        # Record a simple accuracy progress (extend as needed)
        accuracy_progress = [train_accuracy]

        # Create the artifact package with an auto-generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "LogisticRegression",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "accuracy_progress": accuracy_progress,
            "trained_model": model  # This key will be processed and removed in store_model_artifact.
        }

        # Store the artifact using the DatasetModel's store_model_artifact function.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)
        
        logging.info(f"LogisticRegression trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")
        
        return {
            "model_name": "LogisticRegression",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally, you can return the model object as well.
        }
        
    except Exception as e:
        logging.error(f"Error in logistic_regression: {e}")
        raise CustomException(e, sys)
