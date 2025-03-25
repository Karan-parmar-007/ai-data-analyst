import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from app.utils.exception import CustomException
from app.utils.logger import logging
from app.models.datasets import DatasetModel

# --- Classification Models ---

def train_logistic_regression(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a LogisticRegression model.
    
    Hyperparameters (many of which are optional):
      - penalty: str, {'l1', 'l2', 'elasticnet', 'none'} (default 'l2')
      - dual: bool (default False)
      - tol: float (default 1e-4)
      - C: float (default 1.0)
      - fit_intercept: bool (default True)
      - intercept_scaling: float (default 1)
      - class_weight: dict or 'balanced' (default None)
      - random_state: int (default None)
      - solver: str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'} (default 'lbfgs')
      - max_iter: int (default 100)
      - multi_class: str, {'auto', 'ovr', 'multinomial'} (default 'auto')
    """
    try:
        model = LogisticRegression(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted")
        }

        # Store the trained model artifact.
        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("LogisticRegression", model_pickle)

        logging.info(f"LogisticRegression training completed in {train_time:.2f} seconds.")
        return {"model_name": "LogisticRegression", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training LogisticRegression: {e}")
        raise CustomException(e, sys)

def train_random_forest_classifier(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a RandomForestClassifier model.
    
    Hyperparameters:
      - n_estimators: int (default 100)
      - criterion: str, {'gini', 'entropy', 'log_loss'} (default 'gini')
      - max_depth: int (default None)
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_features: {"sqrt", "log2"} or int or float (default "auto")
      - max_leaf_nodes: int (default None)
      - min_impurity_decrease: float (default 0.0)
      - bootstrap: bool (default True)
      - oob_score: bool (default False)
      - n_jobs: int (default None)
      - random_state: int (default None)
      - verbose: int (default 0)
      - warm_start: bool (default False)
      - class_weight: dict or "balanced" (default None)
    """
    try:
        model = RandomForestClassifier(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted")
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("RandomForestClassifier", model_pickle)

        logging.info(f"RandomForestClassifier training completed in {train_time:.2f} seconds.")
        return {"model_name": "RandomForestClassifier", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training RandomForestClassifier: {e}")
        raise CustomException(e, sys)

def train_svc(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a Support Vector Classifier (SVC) model.
    
    Hyperparameters:
      - C: float (default 1.0)
      - kernel: str, {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} (default 'rbf')
      - degree: int (default 3) – used for poly kernel
      - gamma: {'scale', 'auto'} or float (default 'scale')
      - coef0: float (default 0.0)
      - shrinking: bool (default True)
      - probability: bool (default False)
      - tol: float (default 1e-3)
      - cache_size: float (default 200)
      - class_weight: dict or 'balanced' (default None)
      - verbose: bool (default False)
      - max_iter: int (default -1, i.e., no limit)
      - decision_function_shape: str, {'ovo', 'ovr'} (default 'ovr')
      - random_state: int (default None)
    """
    try:
        model = SVC(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted")
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("SVC", model_pickle)

        logging.info(f"SVC training completed in {train_time:.2f} seconds.")
        return {"model_name": "SVC", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training SVC: {e}")
        raise CustomException(e, sys)

def train_decision_tree_classifier(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a DecisionTreeClassifier model.
    
    Hyperparameters:
      - criterion: str, {'gini', 'entropy', 'log_loss'} (default 'gini')
      - splitter: str, {'best', 'random'} (default 'best')
      - max_depth: int (default None)
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_features: int, float, or {"sqrt", "log2"} (default None)
      - random_state: int (default None)
      - max_leaf_nodes: int (default None)
      - class_weight: dict or 'balanced' (default None)
      - ccp_alpha: float (default 0.0)
    """
    try:
        model = DecisionTreeClassifier(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted")
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("DecisionTreeClassifier", model_pickle)

        logging.info(f"DecisionTreeClassifier training completed in {train_time:.2f} seconds.")
        return {"model_name": "DecisionTreeClassifier", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training DecisionTreeClassifier: {e}")
        raise CustomException(e, sys)

def train_knn_classifier(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a KNeighborsClassifier model.
    
    Hyperparameters:
      - n_neighbors: int (default 5)
      - weights: str or callable, {'uniform', 'distance'} (default 'uniform')
      - algorithm: str, {'auto', 'ball_tree', 'kd_tree', 'brute'} (default 'auto')
      - leaf_size: int (default 30)
      - p: int, power parameter for Minkowski metric (default 2)
      - metric: str or callable (default 'minkowski')
      - metric_params: dict (default None)
      - n_jobs: int (default None)
    """
    try:
        model = KNeighborsClassifier(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted")
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("KNeighborsClassifier", model_pickle)

        logging.info(f"KNeighborsClassifier training completed in {train_time:.2f} seconds.")
        return {"model_name": "KNeighborsClassifier", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training KNeighborsClassifier: {e}")
        raise CustomException(e, sys)

# --- Regression Models ---

def train_linear_regression(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a LinearRegression model.
    
    Hyperparameters:
      - fit_intercept: bool (default True)
      - normalize: bool (deprecated in newer versions; use preprocessing instead)
      - copy_X: bool (default True)
      - n_jobs: int (default None)
    """
    try:
        model = LinearRegression(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred)
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("LinearRegression", model_pickle)

        logging.info(f"LinearRegression training completed in {train_time:.2f} seconds.")
        return {"model_name": "LinearRegression", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training LinearRegression: {e}")
        raise CustomException(e, sys)

def train_random_forest_regressor(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a RandomForestRegressor model.
    
    Hyperparameters:
      - n_estimators: int (default 100)
      - criterion: str, {'mse', 'mae', 'poisson'} (default 'mse')
      - max_depth: int (default None)
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_features: {"sqrt", "log2"} or int or float (default "auto")
      - max_leaf_nodes: int (default None)
      - bootstrap: bool (default True)
      - oob_score: bool (default False)
      - random_state: int (default None)
      - n_jobs: int (default None)
      - verbose: int (default 0)
      - warm_start: bool (default False)
    """
    try:
        model = RandomForestRegressor(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred)
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("RandomForestRegressor", model_pickle)

        logging.info(f"RandomForestRegressor training completed in {train_time:.2f} seconds.")
        return {"model_name": "RandomForestRegressor", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training RandomForestRegressor: {e}")
        raise CustomException(e, sys)

def train_svr(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a Support Vector Regressor (SVR) model.
    
    Hyperparameters:
      - kernel: str, {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} (default 'rbf')
      - degree: int (default 3) – used for poly kernel
      - gamma: {'scale', 'auto'} or float (default 'scale')
      - coef0: float (default 0.0)
      - tol: float (default 1e-3)
      - C: float (default 1.0)
      - epsilon: float (default 0.1)
      - shrinking: bool (default True)
      - cache_size: float (default 200)
      - verbose: bool (default False)
      - max_iter: int (default -1, i.e., no limit)
    """
    try:
        model = SVR(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred)
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("SVR", model_pickle)

        logging.info(f"SVR training completed in {train_time:.2f} seconds.")
        return {"model_name": "SVR", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training SVR: {e}")
        raise CustomException(e, sys)

def train_decision_tree_regressor(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a DecisionTreeRegressor model.
    
    Hyperparameters:
      - criterion: str, {'mse', 'friedman_mse', 'mae', 'poisson'} (default 'mse')
      - splitter: str, {'best', 'random'} (default 'best')
      - max_depth: int (default None)
      - min_samples_split: int or float (default 2)
      - min_samples_leaf: int or float (default 1)
      - min_weight_fraction_leaf: float (default 0.0)
      - max_features: int, float, or {"sqrt", "log2"} (default None)
      - random_state: int (default None)
      - max_leaf_nodes: int (default None)
      - ccp_alpha: float (default 0.0)
    """
    try:
        model = DecisionTreeRegressor(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred)
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("DecisionTreeRegressor", model_pickle)

        logging.info(f"DecisionTreeRegressor training completed in {train_time:.2f} seconds.")
        return {"model_name": "DecisionTreeRegressor", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training DecisionTreeRegressor: {e}")
        raise CustomException(e, sys)

def train_knn_regressor(X_train, X_test, y_train, y_test, hyperparameters: dict):
    """
    Train a KNeighborsRegressor model.
    
    Hyperparameters:
      - n_neighbors: int (default 5)
      - weights: str or callable, {'uniform', 'distance'} (default 'uniform')
      - algorithm: str, {'auto', 'ball_tree', 'kd_tree', 'brute'} (default 'auto')
      - leaf_size: int (default 30)
      - p: int, power parameter for Minkowski metric (default 2)
      - metric: str or callable (default 'minkowski')
      - metric_params: dict (default None)
      - n_jobs: int (default None)
    """
    try:
        model = KNeighborsRegressor(**hyperparameters)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred)
        }

        model_pickle = pickle.dumps(model)
        dataset_model = DatasetModel()
        dataset_model.store_artifacts("KNeighborsRegressor", model_pickle)

        logging.info(f"KNeighborsRegressor training completed in {train_time:.2f} seconds.")
        return {"model_name": "KNeighborsRegressor", "train_time": train_time, "metrics": metrics, "trained_model": model}
    except Exception as e:
        logging.error(f"Error training KNeighborsRegressor: {e}")
        raise CustomException(e, sys)
