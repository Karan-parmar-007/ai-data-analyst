from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets import DatasetModel
from datetime import datetime
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
import numpy as np
from bson import ObjectId  # Added import for ObjectId

_scheduler = None

# Initialize DatasetModel
dataset_model = DatasetModel()

# Helper function to load dataset from GridFS
def load_dataset(dataset_id: str) -> pd.DataFrame:
    grid_out = dataset_model.get_dataset_csv(dataset_id)
    dataset_details = dataset_model.get_dataset(dataset_id)
    if dataset_details.get("is_preprocessing_done") and "preprocessed_file_id" in dataset_details:
        grid_out = dataset_model.fs.get(dataset_details["preprocessed_file_id"])
    return pd.read_csv(BytesIO(grid_out.read()))

# Helper function to infer target column
def infer_target_column(dataset: dict, df: pd.DataFrame) -> str:
    """Infer target column from usage_of_each_column or default to last column."""
    usage = dataset.get("usage_of_each_column", {})
    for col, role in usage.items():
        if role.lower() in ["target", "label", "output"]:
            return col
    
    # Fallback: Assume last column is target if no explicit usage
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
def train_model(model_type: str, X_train, X_test, y_train, y_test):
    # Define models and parameter grids
    model_configs = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "logistic_regression": {
            "model": LogisticRegression(max_iter=200),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5]
            }
        },
        "kmeans": {
            "model": KMeans(),
            "params": {
                "n_clusters": [2, 3, 4],
                "max_iter": [100, 300]
            }
        },
        "mlp": {
            "model": MLPClassifier(max_iter=200),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "learning_rate_init": [0.001, 0.01]
            }
        }
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = model_configs[model_type]
    model = config["model"]
    param_grid = config["params"]

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error" if model_type == "linear_regression" else "accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        hyperparameters = grid_search.best_params_
    else:
        model.fit(X_train, y_train)
        hyperparameters = {}

    metrics = {}
    if model_type == "linear_regression":
        y_pred = model.predict(X_test)
        metrics["mse"] = float(np.mean((y_pred - y_test) ** 2))
        metrics["r2"] = float(model.score(X_test, y_test))
    elif model_type == "kmeans":
        metrics["inertia"] = float(model.inertia_)
    else:  # Classification models
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(np.mean(y_pred == y_test))

    return model, metrics, hyperparameters

def train_and_evaluate_models():
    """Cron job to train models and find the best one with enhancements."""
    try:
        datasets = dataset_model.datasets_collection.find({"is_preprocessing_done": True})
        
        for dataset in datasets:
            dataset_id = str(dataset["_id"])
            target_column = dataset.get("target_column") or infer_target_column(dataset, load_dataset(dataset_id))
            
            if not target_column:
                print(f"Skipping dataset {dataset_id}: No target column specified or inferred.")
                continue
            
            df = load_dataset(dataset_id)
            if target_column not in df.columns:
                print(f"Skipping dataset {dataset_id}: Target column '{target_column}' not found.")
                continue
            
            X, y = preprocess_dataset(df, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_types = [
                "linear_regression",
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "kmeans",
                "mlp"
            ]

            best_model = None
            best_metric = float('inf')  # For MSE (regression)
            best_accuracy = -float('inf')  # For accuracy (classification)
            best_job_id = None
            best_hyperparameters = None

            for model_type in model_types:
                try:
                    model, metrics, hyperparameters = train_model(model_type, X_train, X_test, y_train, y_test)
                    
                    job_id = str(ObjectId())
                    model_entry = {
                        "job_id": job_id,
                        "model_type": model_type,
                        "metrics": metrics,
                        "hyperparameters": hyperparameters,
                        "target_column": target_column,
                        "timestamp": datetime.utcnow()
                    }
                    
                    dataset_model.update_dataset(dataset_id, {"$push": {"models": model_entry}})

                    if model_type == "linear_regression":
                        metric_value = metrics.get("mse", float('inf'))
                        if metric_value < best_metric:
                            best_metric = metric_value
                            best_model = model_type
                            best_job_id = job_id
                            best_hyperparameters = hyperparameters
                    elif model_type == "kmeans":
                        continue  # Skip KMeans for best model
                    else:  # Classification
                        metric_value = metrics.get("accuracy", -float('inf'))
                        if metric_value > best_accuracy:
                            best_accuracy = metric_value
                            best_model = model_type
                            best_job_id = job_id
                            best_hyperparameters = hyperparameters
                    
                except Exception as e:
                    print(f"Error training {model_type} on dataset {dataset_id}: {str(e)}")

            if best_model:
                metric_name = "mse" if best_model == "linear_regression" else "accuracy"
                metric_value = best_metric if best_model == "linear_regression" else best_accuracy
                best_model_update = {
                    "best_model": {
                        "job_id": best_job_id,
                        "model_type": best_model,
                        "metric": metric_value,
                        "metric_name": metric_name,
                        "hyperparameters": best_hyperparameters,
                        "timestamp": datetime.utcnow()
                    }
                }
                dataset_model.update_dataset(dataset_id, {"$set": best_model_update})
                print(f"Best model for dataset {dataset_id}: {best_model} with {metric_name}={metric_value}")

    except Exception as e:
        print(f"Error in train_and_evaluate_models: {str(e)}")

def model_init_scheduler(app):
    """Initialize the scheduler with the Flask app context."""
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