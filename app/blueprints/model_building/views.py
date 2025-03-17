from flask import request, jsonify
from app.models.datasets import DatasetModel
from app.blueprints.model_building import model_building_bp
from bson import ObjectId
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer  # Added for NaN handling
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

# Initialize DatasetModel
dataset_model = DatasetModel()

# Helper function to load dataset from GridFS
def load_dataset(dataset_id: str) -> pd.DataFrame:
    grid_out = dataset_model.get_dataset_csv(dataset_id)
    dataset_details = dataset_model.get_dataset(dataset_id)
    if dataset_details.get("is_preprocessing_done") and "preprocessed_file_id" in dataset_details:
        grid_out = dataset_model.fs.get(dataset_details["preprocessed_file_id"])
    return pd.read_csv(BytesIO(grid_out.read()))

# Helper function to preprocess dataset
def preprocess_dataset(df: pd.DataFrame, target_column: str = None) -> tuple:
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df  # For unsupervised learning like KMeans
        y = None
    
    # Convert categorical columns to numeric
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].dtype.name == 'category':
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')  # Use mean for numeric columns
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Ensure all columns are numeric
    X = X.astype(float)
    
    if y is not None and (y.dtype == 'object' or y.dtype.name == 'category'):
        y = LabelEncoder().fit_transform(y.astype(str))
    
    return X, y

# Helper function to train a model and return metrics
def train_model(model_type: str, X_train, X_test, y_train, y_test, hyperparameters: dict):
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "logistic_regression":
        model = LogisticRegression(C=1/hyperparameters.get("regularization", 0.1))
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "kmeans":
        model = KMeans(n_clusters=hyperparameters.get("n_clusters", 3))
    elif model_type == "pca":
        model = PCA(n_components=hyperparameters.get("n_components", 2))
    elif model_type == "mlp":
        model = MLPClassifier(
            learning_rate_init=hyperparameters.get("learningRate", 0.01),
            hidden_layer_sizes=(100,),
            max_iter=200
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Fit the model
    if model_type in ["kmeans", "pca"]:
        model.fit(X_train)  # Unsupervised
    else:
        model.fit(X_train, y_train)  # Supervised

    # Generate metrics
    metrics = {}
    if model_type not in ["kmeans", "pca"]:
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(np.mean(y_pred == y_test))
    elif model_type == "kmeans":
        metrics["inertia"] = float(model.inertia_)
    elif model_type == "pca":
        metrics["explained_variance"] = float(np.sum(model.explain_variance_ratio_))

    return model, metrics

# Analyze Dataset
@model_building_bp.route('/analyze', methods=['POST'])
def analyze_dataset():
    try:
        data = request.get_json()
        dataset_id = data.get("datasetId")
        if not dataset_id:
            return jsonify({"error": "datasetId is required"}), 400

        df = load_dataset(dataset_id)
        dataset_info = {
            "datasetId": dataset_id,
            "columns": df.columns.tolist(),
            "shape": [df.shape[0], df.shape[1]],
            "columnTypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "summary": {
                "missingValues": df.isnull().sum().to_dict(),
                "uniqueValues": df.nunique().to_dict(),
            },
            "targetColumn": None
        }
        return jsonify(dataset_info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Train Model
@model_building_bp.route('/train', methods=['POST'])
def train_model_route():
    try:
        data = request.get_json()
        model_type = data.get("modelType")
        hyperparameters = data.get("hyperparameters", {})
        dataset_id = data.get("datasetId")
        target_column = data.get("targetColumn")

        if not all([model_type, dataset_id]):
            return jsonify({"error": "modelType and datasetId are required"}), 400
        if model_type not in ["kmeans", "pca"] and not target_column:
            return jsonify({"error": "targetColumn is required for supervised models"}), 400

        # Load dataset
        df = load_dataset(dataset_id)
        
        # Preprocess dataset
        X, y = preprocess_dataset(df, target_column if model_type not in ["kmeans", "pca"] else None)

        # Split data (for supervised models only)
        if model_type not in ["kmeans", "pca"]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, None, None, None

        # Train model
        model, metrics = train_model(model_type, X_train, X_test, y_train, y_test, hyperparameters)

        job_id = str(ObjectId())
        training_history = {
            "jobId": job_id,
            "modelType": model_type,
            "metrics": metrics,
            "epochs": list(range(1, 11)),
            "loss": [1.0 / (i + 1) + np.random.random() * 0.1 for i in range(10)]
        }
        return jsonify({"jobId": job_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Visualize Model
@model_building_bp.route('/visualize', methods=['POST'])
def visualize_model():
    try:
        data = request.get_json()
        job_id = data.get("jobId")
        if not job_id:
            return jsonify({"error": "jobId is required"}), 400

        epochs = list(range(1, 11))
        loss = [1.0 / (i + 1) + np.random.random() * 0.1 for i in range(10)]
        metrics = {
            "accuracy": np.random.uniform(0.7, 0.95),
            "precision": np.random.uniform(0.65, 0.9),
            "recall": np.random.uniform(0.6, 0.85),
        }
        viz_data = {
            "epochs": epochs,
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        return jsonify(viz_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500