import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def spectral_clustering(
    dataset_id: str,
    X_train: pd.DataFrame,
    n_clusters=8,
    eigen_solver=None,
    n_components=None,
    random_state=None,
    n_init=10,
    gamma=1.0,
    affinity='rbf',
    n_neighbors=10,
    eigen_tol='auto',
    assign_labels='kmeans',
    degree=3,
    coef0=1,
    kernel_params=None,
    n_jobs=None,
    verbose=False
):
    """
    Trains a SpectralClustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_clusters: int (default 8)
      - eigen_solver: str or None (default None)
      - n_components: int or None (default None)
      - random_state: int or None (default None)
      - n_init: int (default 10)
      - gamma: float (default 1.0)
      - affinity: {'nearest_neighbors', 'rbf', 'precomputed', etc.} (default 'rbf')
      - n_neighbors: int (default 10)
      - eigen_tol: float or str (default 'auto')
      - assign_labels: {'kmeans', 'discretize'} (default 'kmeans')
      - degree: int (default 3)
      - coef0: float (default 1)
      - kernel_params: dict or None (default None)
      - n_jobs: int or None (default None)
      - verbose: bool (default False)
    
    Since SpectralClustering does not have a `fit_predict` method, we use `.fit()` followed by `.labels_`.  
    After training, the model is serialized, and cluster labels are stored as an artifact.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact information.
    """
    try:
        # Initialize the SpectralClustering model
        model = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            n_components=n_components,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            verbose=verbose
        )

        logging.info("Starting training for SpectralClustering.")
        start_time = time.time()
        model.fit(X_train)
        cluster_labels = model.labels_
        train_time = time.time() - start_time

        # Compute Silhouette Score (only if more than one cluster is found)
        silhouette = silhouette_score(X_train, cluster_labels) if len(set(cluster_labels)) > 1 else None

        # Serialize the trained model using pickle
        model_pickle = pickle.dumps(model)

        # Prepare the artifact package
        artifact_package = {
            "model_pickle": model_pickle,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": silhouette
        }

        # Store artifacts in the database using DatasetModel
        dataset_model = DatasetModel()
        store_response = dataset_model.store_artifacts(dataset_id, artifact_package)
        
        logging.info(f"SpectralClustering trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")
        
        return {
            "model_name": "SpectralClustering",
            "train_time": train_time,
            "metrics": {
                "silhouette_score": silhouette
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object
        }
        
    except Exception as e:
        logging.error(f"Error in spectral_clustering: {e}")
        raise CustomException(e, sys)
