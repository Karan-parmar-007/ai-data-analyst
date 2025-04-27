from app.models_for_training.clustering_models.k_means_clustering import k_means_clustering
from app.models_for_training.clustering_models.dbscam import dbscan
from app.models_for_training.clustering_models.agglomerative_clustering import agglomerative_clustering
from app.models_for_training.clustering_models.birch import birch
from app.models_for_training.clustering_models.mean_shift import mean_shift
from app.models_for_training.clustering_models.mini_batch_k_means import mini_batch_k_means
from app.models_for_training.clustering_models.spectral_clustering import spectral_clustering

class ClusteringModelTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "KMeans":
            return k_means_clustering(
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
                random_state=hyperparameters.get('random_state', None)
            )
        elif model_name == "DBSCAN":
            return dbscan(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                eps=hyperparameters.get('eps', 0.5),
                min_samples=hyperparameters.get('min_samples', 5),
                metric=hyperparameters.get('metric', 'euclidean'),
                algorithm=hyperparameters.get('algorithm', 'auto'),
                leaf_size=hyperparameters.get('leaf_size', 30)
            )
        elif model_name == "AgglomerativeClustering":
            return agglomerative_clustering(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_clusters=hyperparameters.get('n_clusters', 2),
                affinity=hyperparameters.get('affinity', 'euclidean'),
                linkage=hyperparameters.get('linkage', 'ward'),
                distance_threshold=hyperparameters.get('distance_threshold', None)
            )
        elif model_name == "Birch":
            return birch(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                threshold=hyperparameters.get('threshold', 0.5),
                branching_factor=hyperparameters.get('branching_factor', 50),
                n_clusters=hyperparameters.get('n_clusters', 3),
                compute_labels=hyperparameters.get('compute_labels', True)
            )
        elif model_name == "MeanShift":
            return mean_shift(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                bandwidth=hyperparameters.get('bandwidth', None),
                seeds=hyperparameters.get('seeds', None),
                bin_seeding=hyperparameters.get('bin_seeding', False),
                min_bin_freq=hyperparameters.get('min_bin_freq', 1)
            )
        elif model_name == "MiniBatchKMeans":
            return mini_batch_k_means(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_clusters=hyperparameters.get('n_clusters', 8),
                init=hyperparameters.get('init', 'k-means++'),
                max_iter=hyperparameters.get('max_iter', 100),
                batch_size=hyperparameters.get('batch_size', 100),
                random_state=hyperparameters.get('random_state', None)
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
                n_init=hyperparameters.get('n_init', 10)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
