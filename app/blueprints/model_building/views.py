from flask import request, jsonify
from app.models.datasets import DatasetModel
from app.blueprints.model_building import model_building_bp
from app.models.model_building import ModelToBeBuilt
from bson import ObjectId
from io import BytesIO
import json
from datetime import datetime
from app.models.classification_model import ClassificationModel
from app.models.model_hyperparameters import ModelHyperparameters

dataset_model = DatasetModel()

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

classification_model = ClassificationModel()
model_hyperparameters = ModelHyperparameters()

@model_building_bp.route('/add_classification', methods=['POST'])
def add_classification():
    data = request.json
    classification_name = data.get("classification_name")
    if not classification_name:
        return jsonify({"error": "Missing classification name"}), 400
    result = classification_model.add_classification(classification_name)
    return jsonify(result)

@model_building_bp.route('/get_classifications', methods=['GET'])
def get_classifications():
    result = classification_model.get_all_classifications()
    return jsonify(result)

@model_building_bp.route('/update_classification/<classification_id>', methods=['PUT'])
def update_classification(classification_id):
    data = request.json
    new_name = data.get("classification_name")
    if not new_name:
        return jsonify({"error": "Missing classification name"}), 400
    result = classification_model.update_classification(classification_id, new_name)
    return jsonify(result)

@model_building_bp.route('/add_model', methods=['POST'])
def add_model():
    data = request.json
    classification_id = data.get("classification_id")
    model_name = data.get("model_name")
    hyperparameters = data.get("hyperparameters", {})
    if not classification_id or not model_name:
        return jsonify({"error": "Missing classification ID or model name"}), 400
    result = classification_model.add_model_to_classification(classification_id, model_name, hyperparameters)
    return jsonify(result)

@model_building_bp.route('/get_all_models', methods=['GET'])
def get_all_models():
    result = classification_model.get_all_classifications_with_models()
    return jsonify(result)

@model_building_bp.route('/get_models/<classification_id>', methods=['GET'])
def get_models_for_classification(classification_id):
    result = classification_model.get_models_for_classification(classification_id)
    return jsonify(result)

@model_building_bp.route('/get_hyperparameters/<model_id>', methods=['GET'])
def get_hyperparameters_for_model(model_id):
    result = model_hyperparameters.get_hyperparameters_for_model(model_id)
    return jsonify(result)

@model_building_bp.route('/update_model_hyperparameters/<model_id>', methods=['PUT'])
def update_model_hyperparameters(model_id):
    data = request.json
    updated_hyperparameters = data.get("hyperparameters")
    if not updated_hyperparameters:
        return jsonify({"error": "Missing hyperparameters"}), 400
    result = model_hyperparameters.update_model_hyperparameters(model_id, updated_hyperparameters)
    return jsonify(result)

@model_building_bp.route('/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    result = classification_model.delete_model(model_id)
    return jsonify(result)

@model_building_bp.route('/delete_classification/<classification_id>', methods=['DELETE'])
def delete_classification(classification_id):
    result = classification_model.delete_classification(classification_id)
    return jsonify(result)
