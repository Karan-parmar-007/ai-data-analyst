import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def knn_classifier(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_neighbors: int = 5,
    weights: str = 'uniform',
    algorithm: str = 'auto',
    leaf_size: int = 30,
    p: int = 2,
    metric: str = 'minkowski',
    metric_params=None,
    n_jobs=None
):
    """
    Trains a KNeighborsClassifier model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_neighbors: int (default 5)
      - weights: str, {'uniform', 'distance'} (default 'uniform')
      - algorithm: str, {'auto', 'ball_tree', 'kd_tree', 'brute'} (default 'auto')
      - leaf_size: int (default 30)
      - p: int, power parameter for the Minkowski metric (default 2)
      - metric: str or callable (default 'minkowski')
      - metric_params: dict or None (default None)
      - n_jobs: int or None (default None)
      
    After training, the model is serialized using pickle along with a record of the training accuracy progress
    (currently a single value representing the final training accuracy). The complete artifact package, which
    includes a generated _id, model name, training time, metrics, and the pickled model, is stored in the dataset's
    'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the KNeighborsClassifier with provided hyperparameters.
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )

        logging.info("Starting training for KNeighborsClassifier.")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate predictions on training and test datasets.
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        precision = precision_score(y_test, y_test_pred, average="weighted")
        recall = recall_score(y_test, y_test_pred, average="weighted")

        # Record a simple accuracy progress (currently, the final training accuracy).
        accuracy_progress = [train_accuracy]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "KNeighborsClassifier",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "accuracy_progress": accuracy_progress,
            "trained_model": model  # This will be pickled and stored separately in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"KNeighborsClassifier trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "KNeighborsClassifier",
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
        logging.error(f"Error in KNeighborsClassifier training: {e}")
        raise CustomException(e, sys)
