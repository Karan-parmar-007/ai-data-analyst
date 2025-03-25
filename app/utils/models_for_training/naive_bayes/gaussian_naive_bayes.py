import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def gaussian_nb(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    priors=None,
    var_smoothing: float = 1e-09
):
    """
    Trains a GaussianNB model using the provided data and hyperparameters.
    
    Hyperparameters:
      - priors: array-like of shape (n_classes,) or None (default None)
      - var_smoothing: float (default 1e-09), portion of the largest variance of all features that is added to variances for calculation stability.
      
    After training, the model is serialized using pickle along with classification metrics including
    training and test accuracy, F1-score, precision, and recall. A progress metric (final training accuracy)
    is also recorded.
    
    The complete artifact package, which includes a generated _id, model name, training time, metrics, and
    the pickled model, is stored in the dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the GaussianNB model with the provided hyperparameters.
        model = GaussianNB(
            priors=priors,
            var_smoothing=var_smoothing
        )

        logging.info("Starting training for GaussianNB.")
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

        # Record a simple progress metric (here, the final training accuracy).
        progress = [train_accuracy]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "GaussianNB",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "progress": progress,
            "trained_model": model  # This will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"GaussianNB trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "GaussianNB",
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
        logging.error(f"Error in GaussianNB training: {e}")
        raise CustomException(e, sys)
