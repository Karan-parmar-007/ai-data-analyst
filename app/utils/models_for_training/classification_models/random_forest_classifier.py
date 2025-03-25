import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def random_forest_classifier(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    criterion: str = 'gini',
    max_depth: int = None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf: float = 0.0,
    max_features: str = 'sqrt',
    max_leaf_nodes: int = None,
    min_impurity_decrease: float = 0.0,
    bootstrap: bool = True,
    oob_score: bool = False,
    n_jobs = None,
    random_state = None,
    verbose: int = 0,
    warm_start: bool = False,
    class_weight=None,
    ccp_alpha: float = 0.0,
    max_samples = None,
    monotonic_cst=None  # Note: monotonic_cst is not used by default in sklearn; include if custom behavior is needed.
):
    """
    Trains a RandomForestClassifier model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_estimators: int (default 100)
      - criterion: str, {'gini', 'entropy'} (default 'gini')
      - max_depth: int (default None)
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_features: str or int or float (default 'sqrt')
      - max_leaf_nodes: int (default None)
      - min_impurity_decrease: float (default 0.0)
      - bootstrap: bool (default True)
      - oob_score: bool (default False)
      - n_jobs: int (default None)
      - random_state: int or None (default None)
      - verbose: int (default 0)
      - warm_start: bool (default False)
      - class_weight: dict or 'balanced' (default None)
      - ccp_alpha: float (default 0.0)
      - max_samples: int, float or None (default None)
      - monotonic_cst: (default None) - custom parameter if needed.
      
    After training, the model is serialized using pickle along with a record of the training accuracy progress
    (currently a single value representing the final training accuracy). The complete artifact package, which 
    includes a generated _id, model name, training time, metrics, and the pickled model, is stored in the dataset's
    'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the RandomForestClassifier with provided hyperparameters.
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
            # Note: monotonic_cst is not part of the default scikit-learn parameters.
        )

        logging.info("Starting training for RandomForestClassifier.")
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

        # Record a simple accuracy progress (currently, the final training accuracy).
        accuracy_progress = [train_accuracy]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "RandomForestClassifier",
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

        logging.info(f"RandomForestClassifier trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "RandomForestClassifier",
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
        logging.error(f"Error in RandomForestClassifier training: {e}")
        raise CustomException(e, sys)
