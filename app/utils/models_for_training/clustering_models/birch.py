import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def birch_clustering(
    dataset_id: str,
    X_train: pd.DataFrame,
    threshold=0.5,
    branching_factor=50,
    n_clusters=3,
    compute_labels=True,
    copy=True  # `copy='deprecated'` in newer versions, setting default to True
):
    """
    Trains a Birch clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - threshold: float (default 0.5) - The radius threshold for subclusters.
      - branching_factor: int (default 50) - Maximum children per node.
      - n_clusters: int or None (default 3) - Number of clusters, None uses subclusters directly.
      - compute_labels: bool (default True) - Whether to compute labels for each sample.
      - copy: bool (default True) - If False, data is used in-place.

    Returns:
        dict: A dictionary containing training time, metrics, and artifact information.
    """
    try:
        # Initialize the Birch model
        model = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=compute_labels,
            copy=copy
        )

        logging.info("Starting training for Birch clustering.")
        start_time = time.time()
        model.fit(X_train)
        cluster_labels = model.labels_
        train_time = time.time() - start_time

        # Compute Silhouette Score (if there is more than one unique cluster)
        silhouette = silhouette_score(X_train, cluster_labels) if len(set(cluster_labels)) > 1 else None

        # Serialize the trained model
        model_pickle = pickle.dumps(model)

        # Prepare the artifact package
        artifact_package = {
            "model_pickle": model_pickle,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": silhouette
        }

        # Store artifacts in the database
        dataset_model = DatasetModel()
        store_response = dataset_model.store_artifacts(dataset_id, artifact_package)

        logging.info(f"Birch clustering trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "Birch",
            "train_time": train_time,
            "metrics": {
                "silhouette_score": silhouette
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object
        }

    except Exception as e:
        logging.error(f"Error in birch_clustering: {e}")
        raise CustomException(e, sys)
