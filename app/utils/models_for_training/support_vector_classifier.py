import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def svc(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    C: float = 1.0,
    kernel: str = 'rbf',
    degree: int = 3,
    gamma: str = 'scale',
    coef0: float = 0.0,
    shrinking: bool = True,
    probability: bool = False,
    tol: float = 0.001,
    cache_size: float = 200,
    class_weight=None,
    verbose: bool = False,
    max_iter: int = -1,
    decision_function_shape: str = 'ovr',
    break_ties: bool = False,
    random_state=None
):
    """
    Trains an SVC model using the provided data and hyperparameters.
    Hyperparameters (from sklearn.svm.SVC):
      - C: float (default 1.0)
      - kernel: str, e.g., 'rbf', 'linear', 'poly', 'sigmoid' (default 'rbf')
      - degree: int (default 3, used when kernel is 'poly')
      - gamma: {'scale', 'auto'} or float (default 'scale')
      - coef0: float (default 0.0)
      - shrinking: bool (default True)
      - probability: bool (default False)
      - tol: float (default 0.001)
      - cache_size: float (default 200)
      - class_weight: dict or 'balanced' (default None)
      - verbose: bool (default False)
      - max_iter: int (default -1, i.e., no limit)
      - decision_function_shape: str, {'ovr', 'ovo'} (default 'ovr')
      - break_ties: bool (default False)
      - random_state: int or None (default None)

    After training, the model is serialized with pickle and an artifact package is created,
    which includes a generated _id, model name, training time, performance metrics, and a
    simple accuracy progress list. This artifact is then stored in the dataset's 'models'
    array in the database.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the SVC model with provided hyperparameters
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )

        logging.info("Starting training for SVC.")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate the model on both training and test datasets.
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        precision = precision_score(y_test, y_test_pred, average="weighted")
        recall = recall_score(y_test, y_test_pred, average="weighted")

        # Record a simple accuracy progress (currently, just the final training accuracy).
        accuracy_progress = [train_accuracy]

        # Create the artifact package with an auto-generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "SVC",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "accuracy_progress": accuracy_progress,
            "trained_model": model  # This will be removed and stored separately in GridFS.
        }

        # Store the artifact using DatasetModel's store_model_artifact function.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"SVC trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "SVC",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object as well.
        }

    except Exception as e:
        logging.error(f"Error in SVC training: {e}")
        raise CustomException(e, sys)
