import sys
import time
import pickle
from bson import ObjectId
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from app.models.datasets import DatasetModel
from app.utils.exception import CustomException
from app.utils.logger import logging

def kmeans(
    dataset_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    # For clustering, targets are not required.
    n_clusters: int = 8,
    init: str = 'k-means++',
    n_init = 'auto',
    max_iter: int = 300,
    tol: float = 0.0001,
    verbose: int = 0,
    random_state = None,
    copy_x: bool = True,
    algorithm: str = 'lloyd'
):
    """
    Trains a KMeans clustering model using the provided data and hyperparameters.
    
    Hyperparameters:
      - n_clusters: int (default 8)
      - init: method for initialization, e.g., 'k-means++' (default 'k-means++')
      - n_init: number of time the k-means algorithm will be run with different centroid seeds (default 'auto')
      - max_iter: maximum number of iterations (default 300)
      - tol: relative tolerance with regards to inertia to declare convergence (default 0.0001)
      - verbose: verbosity mode (default 0)
      - random_state: int or None (default None)
      - copy_x: bool, whether to copy the data (default True)
      - algorithm: algorithm used for computation, e.g., 'lloyd' (default 'lloyd')
      
    Since KMeans is unsupervised, we evaluate performance using the inertia (sum of squared distances to the nearest cluster center)
    on both the training and test data.
    
    After training, the model is serialized using pickle along with the performance metrics and a record of the inertia as progress.
    The complete artifact package, which includes a generated _id, model name, training time, metrics, and the pickled model,
    is stored in the dataset's 'models' array.
    
    Returns:
        dict: A dictionary containing training time, metrics, and artifact storage information.
    """
    try:
        # Initialize the KMeans model with the provided hyperparameters.
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )

        logging.info("Starting training for KMeans clustering.")
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time

        # Evaluate performance using inertia on training and test data.
        train_inertia = model.inertia_
        # For test data, we compute inertia manually.
        test_inertia = np.sum(np.min(
            np.square(X_test.to_numpy() - model.cluster_centers_[:, None].T),
            axis=1
        ))

        # Record a simple progress metric (here, the final training inertia).
        progress = [train_inertia]

        # Create the artifact package with a generated _id.
        artifact_package = {
            "_id": ObjectId(),
            "model_name": "KMeans",
            "train_time": train_time,
            "metrics": {
                "train_inertia": train_inertia,
                "test_inertia": test_inertia
            },
            "progress": progress,
            "trained_model": model  # This will be pickled and stored in GridFS.
        }

        # Store the artifact using the DatasetModel's store_model_artifact method.
        dataset_model = DatasetModel()
        store_response = dataset_model.store_model_artifact(dataset_id, artifact_package)

        logging.info(f"KMeans clustering trained in {train_time:.2f} seconds. Artifacts stored: {store_response}")

        return {
            "model_name": "KMeans",
            "train_time": train_time,
            "metrics": {
                "train_inertia": train_inertia,
                "test_inertia": test_inertia
            },
            "artifact_store_response": store_response,
            "trained_model": model  # Optionally return the model object as well.
        }

    except Exception as e:
        logging.error(f"Error in KMeans training: {e}")
        raise CustomException(e, sys)
