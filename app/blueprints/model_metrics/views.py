from flask import jsonify, request
from app.models.datasets import DatasetModel
from bson import ObjectId
from app.blueprints.model_metrics import model_metrics_bp

dataset_model = DatasetModel()

@model_metrics_bp.route('/metrics', methods=['GET'])
def get_model_metrics():
    try:
        dataset_id = request.args.get("datasetId")
        if not dataset_id:
            return jsonify({"error": "datasetId is required"}), 400

        dataset = dataset_model.get_dataset(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Transform model data to match frontend ModelMetrics interface
        model_metrics = []
        for model in dataset.get("models", []):
            metrics = model["metrics"]
            # Default values for missing metrics (e.g., regression or kmeans)
            model_metrics.append({
                "name": model["model_type"],
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1Score": metrics.get("f1Score", 0.0),
                "trainingTime": metrics.get("trainingTime", 0.0),
                "timestamp": model["timestamp"]
            })

        return jsonify(model_metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500