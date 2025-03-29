import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def gradient_boosting_classifier(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    loss: str = 'log_loss',
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    subsample: float = 1.0,
    criterion: str = 'friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf: float = 0.0,
    max_depth: int = 3,
    min_impurity_decrease: float = 0.0,
    init=None,
    random_state=None,
    max_features=None,
    verbose: int = 0,
    max_leaf_nodes: int = None,
    warm_start: bool = False,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = None,
    tol: float = 0.0001,
    ccp_alpha: float = 0.0
):
    """
    Trains a GradientBoostingClassifier model using the provided data and hyperparameters.
    
    Hyperparameters:
      - loss: str (default 'log_loss')
      - learning_rate: float (default 0.1)
      - n_estimators: int (default 100)
      - subsample: float (default 1.0)
      - criterion: str, (default 'friedman_mse')
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_depth: int (default 3)
      - min_impurity_decrease: float (default 0.0)
      - init: estimator or None (default None)
      - random_state: int or None (default None)
      - max_features: int, float, string or None (default None)
      - verbose: int (default 0)
      - max_leaf_nodes: int or None (default None)
      - warm_start: bool (default False)
      - validation_fraction: float (default 0.1)
      - n_iter_no_change: int or None (default None)
      - tol: float (default 0.0001)
      - ccp_alpha: float (default 0.0)
      
    After training, the model is serialized using pickle along with the training accuracy progress.
    An artifact package (with a generated _id) containing the model name, training time, metrics, and
    the pickled model is stored in the dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the GradientBoostingClassifier with provided hyperparameters.
        model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha
        )

        logging.info("Starting training for GradientBoostingClassifier.")
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

        # Record a simple accuracy progress (currently, just the final training accuracy).
        accuracy_progress = [train_accuracy]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "GradientBoostingClassifier",
            "train_time": train_time,
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            },
            "accuracy_progress": accuracy_progress,
            "trained_model": model  # This will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"GradientBoostingClassifier trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "GradientBoostingClassifier",
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
        logging.error(f"Error in GradientBoostingClassifier training: {e}")
        raise CustomException(e, sys)
