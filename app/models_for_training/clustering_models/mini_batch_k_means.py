import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def mini_batch_kmeans(
    dataset_id: str,
    X_train: pd.DataFrame,
    n_clusters: int = 8,
    init: str = 'k-means++',
    max_iter: int = 100,
    batch_size: int = 1024,
    verbose: int = 0,
    compute_labels: bool = True,
    random_state=None,
    tol: float = 0.0,
    max_no_improvement: int = 10,
    init_size=None,
    n_init: str = 'auto',
    reassignment_ratio: float = 0.01
):
    """
    Trains a MiniBatchKMeans clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_clusters: int (default 8)
      - init: {'k-means++', 'random'} (default 'k-means++')
      - max_iter: int (default 100)
      - batch_size: int (default 1024)
      - verbose: int (default 0)
      - compute_labels: bool (default True)
      - random_state: int or None (default None)
      - tol: float (default 0.0)
      - max_no_improvement: int (default 10)
      - init_size: int or None (default None)
      - n_init: {'auto', int} (default 'auto')
      - reassignment_ratio: float (default 0.01)
    
    After training, the model is serialized using pickle along with a record of cluster assignments.
    
    The artifacts are then stored in the database.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact information.
    """
    try:
        # Initialize the MiniBatchKMeans model with given hyperparameters
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio
        )

        logging.info("Starting training for MiniBatchKMeans.")
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time

        # Compute cluster labels
        cluster_labels = model.labels_
        
        # Compute Silhouette Score as a clustering metric (only if n_clusters > 1)
        silhouette = silhouette_score(X_train, cluster_labels) if n_clusters > 1 else None

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
        
        logging.info(f"MiniBatchKMeans trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")
        
        return {
            "model_name": "MiniBatchKMeans",
            "train_time": train_time,
            "metrics": {
                "silhouette_score": silhouette
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object
        }
        
    except Exception as e:
        logging.error(f"Error in mini_batch_kmeans: {e}")
        raise CustomException(e, sys)
