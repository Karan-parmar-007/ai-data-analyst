from app.blueprints.admin import admin_bp
from flask import Flask, request, jsonify, send_file
from bson import ObjectId, errors
from app.utils.db import db
import pandas as pd
import sklearn
import gridfs
from io import BytesIO
from app.models.datasets import DatasetModel
from app.models.users import UserModel
import io
from flask import Response, stream_with_context
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.models.classification_model import ClassificationModel
from app.models.model_hyperparameters import ModelHyperparameters



# Initialize DatasetManager
dataset_model = DatasetModel()
user_model = UserModel()
classification_model = ClassificationModel()
model_hyperparameters = ModelHyperparameters()

# 1. Add a new classification
@admin_bp.route('/add_classification', methods=['POST'])
def add_classification():
    data = request.json
    classification_name = data.get("classification_name")
    if not classification_name:
        return jsonify({"error": "Missing classification_name"}), 400
    
    result = classification_model.add_classification(classification_name)
    return jsonify(result)

# 2. Get all classifications
@admin_bp.route('/get_classifications', methods=['GET'])
def get_classifications():
    result = classification_model.get_all_classifications()
    return jsonify(result)

# 3. Update classification name
@admin_bp.route('/update_classification/<classification_id>', methods=['PUT'])
def update_classification(classification_id):
    data = request.json
    new_name = data.get("classification_name")
    if not new_name:
        return jsonify({"error": "Missing new classification_name"}), 400

    result = classification_model.update_classification(classification_id, new_name)
    return jsonify(result)

# 4. Add model to a classification
@admin_bp.route('/add_model', methods=['POST'])
def add_model():
    data = request.json
    classification_id = data.get("classification_id")
    model_name = data.get("model_name")
    hyperparameters = data.get("hyperparameters", {})

    if not classification_id or not model_name:
        return jsonify({"error": "Missing classification_id or model_name"}), 400

    result = classification_model.add_model_to_classification(classification_id, model_name, hyperparameters)
    return jsonify(result)

# 5. Get all classifications with their models and hyperparameters
@admin_bp.route('/get_all_models', methods=['GET'])
def get_all_models():
    result = classification_model.get_all_classifications_with_models()
    return jsonify(result)

# 6. Get all models for a specific classification
@admin_bp.route('/get_models/<classification_id>', methods=['GET'])
def get_models_for_classification(classification_id):
    result = classification_model.get_models_for_classification(classification_id)
    return jsonify(result)

# 7. Delete a model
@admin_bp.route('/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    result = classification_model.delete_model(model_id)
    return jsonify(result)

# 8. Delete a classification (and all models within)
@admin_bp.route('/delete_classification/<classification_id>', methods=['DELETE'])
def delete_classification(classification_id):
    result = classification_model.delete_classification(classification_id)
    return jsonify(result)


# 9. Get hyperparameters for a model
@admin_bp.route('/get_hyperparameters/<model_id>', methods=['GET'])
def get_hyperparameters_for_model(model_id):
    result = model_hyperparameters.get_hyperparameters_for_model(model_id)
    return jsonify(result)

# 10. Update hyperparameters for a model
@admin_bp.route('/update_model_hyperparameters/<model_id>', methods=['PUT'])
def update_model_hyperparameters(model_id):
    data = request.json
    updated_hyperparameters = data.get("hyperparameters")
    if not updated_hyperparameters:
        return jsonify({"error": "Missing hyperparameters"}), 400

    result = model_hyperparameters.update_model_hyperparameters(model_id, updated_hyperparameters)
    return jsonify(result)