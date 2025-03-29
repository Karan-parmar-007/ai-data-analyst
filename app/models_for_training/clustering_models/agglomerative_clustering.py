import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def agglomerative_clustering(
    dataset_id: str,
    X_train: pd.DataFrame,
    n_clusters=2,
    metric='euclidean',
    memory=None,
    connectivity=None,
    compute_full_tree='auto',
    linkage='ward',
    distance_threshold=None,
    compute_distances=False
):
    """
    Trains an Agglomerative Clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_clusters: int (default 2)
      - metric: str (default 'euclidean')
      - memory: str or object (default None)
      - connectivity: array-like or callable (default None)
      - compute_full_tree: str or bool (default 'auto')
      - linkage: {'ward', 'complete', 'average', 'single'} (default 'ward')
      - distance_threshold: float or None (default None)
      - compute_distances: bool (default False)

    Since AgglomerativeClustering is a hierarchical algorithm, it does not use a `fit_predict` method.  
    After training, the model is serialized, and cluster labels are stored as an artifact.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact information.
    """
    try:
        # Initialize the AgglomerativeClustering model
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances
        )

        logging.info("Starting training for AgglomerativeClustering.")
        start_time = time.time()
        cluster_labels = model.fit_predict(X_train)
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
        
        logging.info(f"AgglomerativeClustering trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")
        
        return {
            "model_name": "AgglomerativeClustering",
            "train_time": train_time,
            "metrics": {
                "silhouette_score": silhouette
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object
        }
        
    except Exception as e:
        logging.error(f"Error in agglomerative_clustering: {e}")
        raise CustomException(e, sys)
