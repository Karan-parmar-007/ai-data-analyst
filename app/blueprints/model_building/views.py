from flask import request, jsonify
from app.models.datasets import DatasetModel
from app.blueprints.model_building import model_building_bp
from app.models.model_building import ModelToBeBuilt
from app.utils.gemini import GeminiFunctions
from bson import ObjectId
from io import BytesIO
import json
from datetime import datetime
from app.models.classification_model import ClassificationModel
from app.models.model_hyperparameters import ModelHyperparameters

dataset_model = DatasetModel()
classification_model = ClassificationModel()
model_hyperparameters = ModelHyperparameters()

@model_building_bp.route('/start-building', methods=['post'])
def start_building():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return jsonify({"error": "datasetId is required"}), 400
        model_to_be_build = ModelToBeBuilt()
        model_to_be_build.create_model_building_model(dataset_id)
        return jsonify({"message": "Model building started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

    
@model_building_bp.route('/get-preferred-model', methods=['GET'])
def get_preferred_model():
    gemini_function = GeminiFunctions()
    dataset_id = request.args.get('dataset_id')
    model_recommendations = gemini_function.recommend_models_for_dataset(dataset_id)

    if not model_recommendations:
        return jsonify({"error": "No preferred model found"}), 404

    return jsonify(model_recommendations), 200

@model_building_bp.route('/get-model-hyperparameters', methods=['GET'])
def get_model_hyperparameters():
    dataset_id = request.args.get('dataset_id')
    model_id = request.args.get('model_id')
    if not dataset_id or not model_id:
        return jsonify({"error": "dataset_id and model_name are required"}), 400
    gemini_function = GeminiFunctions()
    hyperparameters = gemini_function.get_hyperparameters(dataset_id, model_id)
    if not hyperparameters:
        return jsonify({"error": "No hyperparameters found"}), 404

    return jsonify(hyperparameters), 200
