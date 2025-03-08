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
import numpy as np
import json

# model_building = Blueprint('model_building', __name__, url_prefix='/api')

# Initialize DatasetModel
dataset_model = DatasetModel()

# Helper function to load dataset from GridFS
def load_dataset(dataset_id: str) -> pd.DataFrame:
    grid_out = dataset_model.get_dataset_csv(dataset_id)
    return pd.read_csv(BytesIO(grid_out.read()))

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
        model = KMeans(n_clusters=3)  # Example; adjust as needed
    elif model_type == "pca":
        model = PCA(n_components=2)  # Example
    elif model_type == "mlp":
        model = MLPClassifier(
            learning_rate_init=hyperparameters.get("learningRate", 0.01),
            hidden_layer_sizes=(100,),  # Example
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
    if model_type not in ["kmeans", "pca"]:  # Supervised models
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(np.mean(y_pred == y_test))
        # Add precision, recall, etc., as needed
    elif model_type == "kmeans":
        metrics["inertia"] = float(model.inertia_)
    elif model_type == "pca":
        metrics["explained_variance"] = float(np.sum(model.explained_variance_ratio_))

    return model, metrics

# Analyze Dataset
@model_building_bp.route('/analyze', methods=['POST'])
def analyze_dataset():
    try:
        data = request.get_json()
        dataset_id = data.get("datasetId")
        if not dataset_id:
            return jsonify({"error": "datasetId is required"}), 400

        # Load dataset
        df = load_dataset(dataset_id)

        # Analyze dataset
        dataset_info = {
            "datasetId": dataset_id,
            "columns": df.columns.tolist(),
            "shape": [df.shape[0], df.shape[1]],
            "columnTypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "summary": {
                "missingValues": df.isnull().sum().to_dict(),
                "uniqueValues": df.nunique().to_dict(),
            },
            "targetColumn": None  # Let user select this in frontend
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

        if not all([model_type, dataset_id, target_column]):
            return jsonify({"error": "modelType, datasetId, and targetColumn are required"}), 400

        # Load dataset
        df = load_dataset(dataset_id)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model, metrics = train_model(model_type, X_train, X_test, y_train, y_test, hyperparameters)

        # Simulate training history (e.g., loss curve)
        job_id = str(ObjectId())  # Unique ID for this training job
        training_history = {
            "jobId": job_id,
            "modelType": model_type,
            "metrics": metrics,
            "epochs": list(range(1, 11)),  # Simulated; replace with real training epochs if available
            "loss": [1.0 / (i + 1) + np.random.random() * 0.1 for i in range(10)]  # Simulated loss
        }

        # Optionally store model/training info in MongoDB (not implemented here for simplicity)
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

        # Simulate fetching training results (replace with actual storage if implemented)
        # For now, we'll generate dummy data based on job_id
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