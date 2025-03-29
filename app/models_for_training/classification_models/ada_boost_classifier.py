import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def adaboost_classifier(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    estimator=None,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    algorithm: str = 'deprecated',
    random_state=None
):
    """
    Trains an AdaBoostClassifier model using the provided data and hyperparameters.
    
    Hyperparameters:
      - estimator: Base estimator from which the boosted ensemble is built (default None, which means DecisionTreeClassifier is used)
      - n_estimators: int (default 50)
      - learning_rate: float (default 1.0)
      - algorithm: str (default 'deprecated')
      - random_state: int or None (default None)
      
    After training, the model is serialized using pickle along with classification metrics including
    training and test accuracy, F1-score, precision, and recall.
    
    The complete artifact package, which includes a generated _id, model name, training time, metrics, 
    and the pickled model, is stored in the dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the AdaBoostClassifier with the provided hyperparameters.
        model = AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )

        logging.info("Starting training for AdaBoostClassifier.")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate model predictions on training and test datasets.
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        precision = precision_score(y_test, y_test_pred, average="weighted")
        recall = recall_score(y_test, y_test_pred, average="weighted")

        # Record a simple accuracy progress (here, just the final training accuracy).
        progress = [train_accuracy]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "AdaBoostClassifier",
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

        logging.info(f"AdaBoostClassifier trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "AdaBoostClassifier",
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
        logging.error(f"Error in AdaBoostClassifier training: {e}")
        raise CustomException(e, sys)
