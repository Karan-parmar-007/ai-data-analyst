import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def dbscan(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean',
    metric_params=None,
    algorithm: str = 'auto',
    leaf_size: int = 30,
    p=None,
    n_jobs=None
):
    """
    Trains a DBSCAN clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - eps: float (default 0.5), the maximum distance between two samples for one to be considered as in the neighborhood of the other.
      - min_samples: int (default 5), the number of samples in a neighborhood for a point to be considered as a core point.
      - metric: string (default 'euclidean'), the distance metric to use.
      - metric_params: dict or None (default None), additional keyword arguments for the metric function.
      - algorithm: str (default 'auto'), the algorithm to be used by the NearestNeighbors module.
      - leaf_size: int (default 30), leaf size passed to BallTree or KDTree.
      - p: int or None (default None), the power parameter for the Minkowski metric.
      - n_jobs: int or None (default None), the number of parallel jobs to run.
    
    Since DBSCAN is unsupervised, we evaluate its performance on the training data by:
      - Counting the number of clusters found (excluding noise, which is labeled as -1).
      - Counting the number of noise points.
      - Computing the silhouette score (if there are at least two clusters).
    
    After training, the model is packaged (along with training time and metrics) into an artifact (with a generated _id) and stored
    in the dataset's 'models' array via the DatasetModel's store_model_artifact method.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the DBSCAN model with the provided hyperparameters.
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs
        )

        logging.info("Starting training for DBSCAN clustering.")
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time

        # Obtain clustering labels for training data.
        train_labels = model.labels_

        # Compute the number of clusters (excluding noise labeled as -1)
        cluster_count = len(set(train_labels)) - (1 if -1 in train_labels else 0)
        noise_count = list(train_labels).count(-1)
        
        # Compute silhouette score if there are at least 2 clusters.
        if cluster_count > 1:
            sil_score = silhouette_score(X_train, train_labels)
        else:
            sil_score = None

        # Optionally, process test data with a new DBSCAN instance (since DBSCAN doesn't support predict)
        test_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs
        )
        test_labels = test_model.fit_predict(X_test)
        test_cluster_count = len(set(test_labels)) - (1 if -1 in test_labels else 0)
        test_noise_count = list(test_labels).count(-1)
        if test_cluster_count > 1:
            test_sil_score = silhouette_score(X_test, test_labels)
        else:
            test_sil_score = None

        # Record a simple progress metric (e.g., the number of clusters found in training data).
        progress = [cluster_count]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "DBSCAN",
            "train_time": train_time,
            "metrics": {
                "train_cluster_count": cluster_count,
                "train_noise_count": noise_count,
                "train_silhouette_score": sil_score,
                "test_cluster_count": test_cluster_count,
                "test_noise_count": test_noise_count,
                "test_silhouette_score": test_sil_score
            },
            "progress": progress,
            "trained_model": model  # This model will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"DBSCAN clustering trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "DBSCAN",
            "train_time": train_time,
            "metrics": {
                "train_cluster_count": cluster_count,
                "train_noise_count": noise_count,
                "train_silhouette_score": sil_score,
                "test_cluster_count": test_cluster_count,
                "test_noise_count": test_noise_count,
                "test_silhouette_score": test_sil_score
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object as well.
        }

    except Exception as e:
        logging.error(f"Error in DBSCAN training: {e}")
        raise CustomException(e, sys)
