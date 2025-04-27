from app.models_for_training.clustering_models.k_means_clustering import kmeans
from app.models_for_training.clustering_models.dbscam import dbscan
from app.models_for_training.clustering_models.agglomerative_clustering import agglomerative_clustering
from app.models_for_training.clustering_models.birch import birch_clustering
from app.models_for_training.clustering_models.mean_shift import mean_shift_clustering
from app.models_for_training.clustering_models.mini_batch_k_means import mini_batch_kmeans
from app.models_for_training.clustering_models.spectral_clustering import spectral_clustering
import numpy as np

class ClusteringModelTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "KMeans":
            return kmeans(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_clusters=hyperparameters.get('n_clusters', 8),
                init=hyperparameters.get('init', 'k-means++'),
                n_init=hyperparameters.get('n_init', 10),
                max_iter=hyperparameters.get('max_iter', 300),
                tol=hyperparameters.get('tol', 1e-4),
                verbose=hyperparameters.get('verbose', 0),
                random_state=hyperparameters.get('random_state', None),
                copy_x=hyperparameters.get('copy_x', True),
                algorithm=hyperparameters.get('algorithm', 'lloyd'),
            )
        elif model_name == "DBSCAN":
            return dbscan(
                self.dataset_id,
                self.X_train,
                self.X_test,
                eps=hyperparameters.get('eps', 0.5),
                min_samples=hyperparameters.get('min_samples', 5),
                metric=hyperparameters.get('metric', 'euclidean'),
                metric_params=hyperparameters.get('metric_params', None),
                algorithm=hyperparameters.get('algorithm', 'auto'),
                leaf_size=hyperparameters.get('leaf_size', 30),
                p=hyperparameters.get('p', None),
                n_jobs=hyperparameters.get('n_jobs', None)
            )
        elif model_name == "AgglomerativeClustering":
            distance_threshold = hyperparameters.get('distance_threshold', None)
            if isinstance(distance_threshold, list):
                # Extract the first valid float value from the list or set to None
                distance_threshold = next((x for x in distance_threshold if isinstance(x, float)), None)
            elif not (distance_threshold is None or isinstance(distance_threshold, float)):
                distance_threshold = None

            # Debug: Log the hyperparameters to verify the linkage value
            print(f"Debug: Hyperparameters for AgglomerativeClustering: {hyperparameters}")

            linkage = hyperparameters.get('linkage', 'ward')
            if isinstance(linkage, list):
                # Extract the first valid string value from the list or set to default 'ward'
                linkage = next((x for x in linkage if isinstance(x, str) and x in {'ward', 'average', 'single', 'complete'}), 'ward')
            elif linkage not in {'ward', 'average', 'single', 'complete'}:
                linkage = 'ward'

            # Validate and convert the `n_clusters` parameter
            n_clusters = hyperparameters.get('n_clusters', 2)
            if isinstance(n_clusters, str) and n_clusters.isdigit():
                n_clusters = int(n_clusters)
            elif not isinstance(n_clusters, int):
                n_clusters = 2  # Default value

            return agglomerative_clustering(
                dataset_id=self.dataset_id,
                X_train=self.X_train,
                n_clusters=n_clusters,  # Pass validated parameter
                metric=hyperparameters.get('affinity', 'euclidean'),
                linkage=linkage,
                distance_threshold=distance_threshold
            )
        elif model_name == "Birch":
            return birch_clustering(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                threshold=hyperparameters.get('threshold', 0.5),
                branching_factor=hyperparameters.get('branching_factor', 50),
                n_clusters=hyperparameters.get('n_clusters', 3),
                compute_labels=np.bool_(hyperparameters.get('compute_labels', True)),
                copy=np.bool_(hyperparameters.get('copy', True))

            )
        elif model_name == "MeanShift":
            return mean_shift_clustering(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                bandwidth=hyperparameters.get('bandwidth', None),
                seeds=hyperparameters.get('seeds', None),
                bin_seeding=np.bool_(hyperparameters.get('bin_seeding', False)),
                min_bin_freq=hyperparameters.get('min_bin_freq', 1),
                cluster_all=np.bool_(hyperparameters.get('cluster_all', True)),
                n_jobs=hyperparameters.get('n_jobs', None),
                max_iter=hyperparameters.get('max_iter', 300),
            )
        elif model_name == "MiniBatchKMeans":
            return mini_batch_kmeans(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_clusters=hyperparameters.get('n_clusters', 8),
                init=hyperparameters.get('init', 'k-means++'),
                max_iter=hyperparameters.get('max_iter', 100),
                batch_size=hyperparameters.get('batch_size', 100),
                verbose=hyperparameters.get('verbose', 0),
                compute_labels=np.bool_(hyperparameters.get('compute_labels', True)),
                random_state=hyperparameters.get('random_state', None),
                tol=hyperparameters.get('tol', 0.0),
                max_no_improvement=hyperparameters.get('max_no_improvement', 10),
                init_size=hyperparameters.get('init_size', None),
                n_init=hyperparameters.get('n_init', 'auto'),
                reassignment_ratio=hyperparameters.get('reassignment_ratio', 0.01)
            )
        elif model_name == "SpectralClustering":
            return spectral_clustering(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_clusters=hyperparameters.get('n_clusters', 8),
                eigen_solver=hyperparameters.get('eigen_solver', None),
                n_components=hyperparameters.get('n_components', None),
                random_state=hyperparameters.get('random_state', None),
                n_init=hyperparameters.get('n_init', 10),
                gamma=hyperparameters.get('gamma', 1.0),
                affinity=hyperparameters.get('affinity', 'rbf'),
                n_neighbors=hyperparameters.get('n_neighbors', 10),
                eigen_tol=hyperparameters.get('eigen_tol', 'auto'),
                assign_labels=hyperparameters.get('assign_labels', 'kmeans'),
                degree=hyperparameters.get('degree', 3),
                coef0=hyperparameters.get('coef0', 1),
                kernel_params=hyperparameters.get('kernel_params', None),
                n_jobs=hyperparameters.get('n_jobs', None),
                verbose=hyperparameters.get('verbose', False)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
