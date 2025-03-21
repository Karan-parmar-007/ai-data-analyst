from apscheduler.schedulers.background import BackgroundScheduler
import os
import hashlib
import time
from app.models.datasets import DatasetModel
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from bson import ObjectId

_scheduler = None
dataset_model = DatasetModel()

# Thresholds for model performance
MIN_ACCEPTABLE_ACCURACY = 0.75  # For classification
MAX_ACCEPTABLE_MSE = 0.1        # For regression

# Helper function to compute dataset hash
def compute_dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# Helper function to load dataset from GridFS (aligned with DatasetModel)
def load_dataset(dataset_id: str) -> pd.DataFrame:
    grid_out = dataset_model.get_dataset_csv(dataset_id)
    return pd.read_csv(BytesIO(grid_out.read()))

# Helper function to infer target column
def infer_target_column(dataset: dict, df: pd.DataFrame) -> str:
    usage = dataset.get("usage_of_each_column", {})
    for col, role in usage.items():
        if role.lower() in ["target", "label", "output"]:
            return col
    return df.columns[-1] if df.columns.size > 1 else None

# Helper function to preprocess dataset
def preprocess_dataset(df: pd.DataFrame, target_column: str) -> tuple:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].dtype.name == 'category':
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = X.astype(float)
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
    
    return X, y

# Helper function to train a model with grid search and return metrics
def train_model(model_type: str, X_train, X_test, y_train, y_test, lightweight=True):
    model_configs = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "logistic_regression": {
            "model": LogisticRegression(max_iter=200),
            "params": {"C": [0.1, 1.0]} if lightweight else {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(),
            "params": {"max_depth": [10]} if lightweight else {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}
        },
        "random_forest": {
            "model": RandomForestClassifier(),
            "params": {"n_estimators": [50], "max_depth": [10]} if lightweight else {"n_estimators": [50, 100], "max_depth": [None, 10], "min_samples_split": [2, 5]}
        },
        "kmeans": {
            "model": KMeans(),
            "params": {"n_clusters": [2], "max_iter": [100]} if lightweight else {"n_clusters": [2, 3, 4], "max_iter": [100, 300]}
        },
        "mlp": {
            "model": MLPClassifier(max_iter=200),
            "params": {"hidden_layer_sizes": [(50,)]} if lightweight else {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "learning_rate_init": [0.001, 0.01]}
        }
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = model_configs[model_type]
    model = config["model"]
    param_grid = config["params"]

    start_time = time.time()
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=3, 
                                 scoring="neg_mean_squared_error" if model_type == "linear_regression" else "accuracy", 
                                 n_jobs=1 if lightweight else 2)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        hyperparameters = grid_search.best_params_
    else:
        model.fit(X_train, y_train)
        hyperparameters = {}
    metrics = {"training_time": time.time() - start_time}

    if model_type == "linear_regression":
        y_pred = model.predict(X_test)
        metrics["mse"] = float(np.mean((y_pred - y_test) ** 2))
        metrics["r2"] = float(model.score(X_test, y_test))
    elif model_type == "kmeans":
        metrics["inertia"] = float(model.inertia_)
    else:
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(np.mean(y_pred == y_test))
        metrics["precision"] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics["f1Score"] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    return model, metrics, hyperparameters

def should_retrain(dataset: dict, df: pd.DataFrame) -> bool:
    """Determine if retraining is needed based on dataset changes or model performance."""
    current_hash = compute_dataset_hash(df)
    last_hash = dataset.get("dataset_hash")
    best_model = dataset.get("best_model", {})
    last_trained = best_model.get("timestamp")

    # Check if dataset has changed
    if last_hash != current_hash:
        return True

    # Check if recently trained (within last 24 hours)
    if last_trained:
        last_trained_time = datetime.fromisoformat(last_trained.replace("Z", "+00:00"))
        if datetime.utcnow() - last_trained_time < timedelta(hours=24):
            return False

    # Check if best model meets performance threshold
    if best_model:
        if best_model.get("metric_name") == "mse" and best_model.get("metric", float('inf')) <= MAX_ACCEPTABLE_MSE:
            return False
        if best_model.get("metric_name") == "accuracy" and best_model.get("metric", 0) >= MIN_ACCEPTABLE_ACCURACY:
            return False
    
    return True

def train_and_evaluate_models():
    """Optimized cron job to train models intelligently."""
    try:
        datasets = list(dataset_model.datasets_collection.find({"is_preprocessing_done": True}))
        if not datasets:
            print("No preprocessed datasets found.")
            return

        for dataset in datasets:
            dataset_id = str(dataset["_id"])
            df = load_dataset(dataset_id)
            target_column = dataset.get("target_column") or infer_target_column(dataset, df)
            
            if not target_column or target_column not in df.columns:
                print(f"Skipping dataset {dataset_id}: Invalid target column.")
                continue

            # Check if retraining is necessary
            if not should_retrain(dataset, df):
                print(f"Skipping dataset {dataset_id}: No retraining needed.")
                continue

            X, y = preprocess_dataset(df, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_types = ["linear_regression", "logistic_regression", "decision_tree"]
            best_model_entry = None
            best_metric = float('inf')  # MSE
            best_accuracy = -float('inf')  # Accuracy

            # Initial lightweight pass
            for model_type in model_types:
                try:
                    model, metrics, hyperparameters = train_model(model_type, X_train, X_test, y_train, y_test, lightweight=True)
                    job_id = str(ObjectId())
                    model_entry = {
                        "dataset_id": ObjectId(dataset_id),
                        "job_id": job_id,
                        "model_type": model_type,
                        "metrics": metrics,
                        "hyperparameters": hyperparameters,
                        "target_column": target_column,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    # Use database directly for model_metrics
                    dataset_model.datasets_collection.database.modelmetrics.insert_one(model_entry)

                    if model_type == "linear_regression":
                        if metrics["mse"] < best_metric:
                            best_metric = metrics["mse"]
                            best_model_entry = model_entry
                    else:
                        if metrics["accuracy"] > best_accuracy:
                            best_accuracy = metrics["accuracy"]
                            best_model_entry = model_entry
                except Exception as e:
                    print(f"Error training {model_type} on dataset {dataset_id}: {str(e)}")

            # Escalate to heavier models if needed
            if (best_model_entry and 
                ((best_model_entry["model_type"] == "linear_regression" and best_metric > MAX_ACCEPTABLE_MSE) or 
                 (best_model_entry["model_type"] != "linear_regression" and best_accuracy < MIN_ACCEPTABLE_ACCURACY))):
                heavy_models = ["random_forest", "mlp"]
                for model_type in heavy_models:
                    try:
                        model, metrics, hyperparameters = train_model(model_type, X_train, X_test, y_train, y_test, lightweight=False)
                        job_id = str(ObjectId())
                        model_entry = {
                            "dataset_id": ObjectId(dataset_id),
                            "job_id": job_id,
                            "model_type": model_type,
                            "metrics": metrics,
                            "hyperparameters": hyperparameters,
                            "target_column": target_column,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        dataset_model.datasets_collection.database.model_metrics.insert_one(model_entry)

                        if metrics["accuracy"] > best_accuracy:
                            best_accuracy = metrics["accuracy"]
                            best_model_entry = model_entry
                    except Exception as e:
                        print(f"Error training {model_type} on dataset {dataset_id}: {str(e)}")

            # Update dataset with best model and metadata
            if best_model_entry:
                metric_name = "mse" if best_model_entry["model_type"] == "linear_regression" else "accuracy"
                metric_value = best_metric if metric_name == "mse" else best_accuracy
                update_fields = {
                    "$set": {
                        "best_model": {
                            "job_id": best_model_entry["job_id"],
                            "model_type": best_model_entry["model_type"],
                            "metric": metric_value,
                            "metric_name": metric_name,
                            "hyperparameters": best_model_entry["hyperparameters"],
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "dataset_hash": compute_dataset_hash(df),
                        "last_trained": datetime.utcnow().isoformat()
                    }
                }
                dataset_model.update_dataset(dataset_id, update_fields)
                print(f"Best model for dataset {dataset_id}: {best_model_entry['model_type']} with {metric_name}={metric_value}")

            time.sleep(1)  # Rate limiting

    except Exception as e:
        print(f"Error in train_and_evaluate_models: {str(e)}")

def model_init_scheduler(app):
    global _scheduler
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        func=train_and_evaluate_models,
        trigger='interval',
        minutes=30,
        id='train_models_job',
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now()
    )
    _scheduler.start()
    import atexit
    atexit.register(lambda: _scheduler.shutdown(wait=False))