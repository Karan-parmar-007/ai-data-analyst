import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def mean_shift_clustering(
    dataset_id: str,
    X_train: pd.DataFrame,
    bandwidth=None,
    seeds=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    n_jobs=None,
    max_iter=300
):
    """
    Trains a MeanShift clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - bandwidth: float or None (default None)
      - seeds: array-like of shape (n_samples, n_features) or None (default None)
      - bin_seeding: bool (default False)
      - min_bin_freq: int (default 1)
      - cluster_all: bool (default True)
      - n_jobs: int or None (default None)
      - max_iter: int (default 300)
    
    After training, the model is serialized using pickle along with a record of cluster assignments.
    
    The artifacts are then stored in the database.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact information.
    """
    try:
        # Initialize the MeanShift model with given hyperparameters
        model = MeanShift(
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter
        )

        logging.info("Starting training for MeanShift.")
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time

        # Compute cluster labels
        cluster_labels = model.labels_
        
        # Compute Silhouette Score as a clustering metric (only if more than one cluster is found)
        silhouette = silhouette_score(X_train, cluster_labels) if len(set(cluster_labels)) > 1 else None

        # Serialize the trained model using pickle
        model_pickle = pickle.dumps(model)

        # Prepare the artifact package
        artifact_package = {
            "model_pickle": model_pickle,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": silhouette
        }

        # Store artifacts in the database using DatasetModel.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_artifacts(dataset_id, artifact_package)
        
        logging.info(f"MeanShift trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")
        
        return {
            "model_name": "MeanShift",
            "train_time": train_time,
            "metrics": {
                "silhouette_score": silhouette
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object
        }
        
    except Exception as e:
        logging.error(f"Error in mean_shift_clustering: {e}")
        raise CustomException(e, sys)
