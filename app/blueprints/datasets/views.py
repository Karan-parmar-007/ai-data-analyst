from app.blueprints.datasets import datasets_bp
from flask import Flask, request, jsonify, send_file
from bson import ObjectId, errors
from app.utils.db import db
import gridfs
from io import BytesIO
from app.models.datasets import DatasetModel
from app.models.users import UserModel
import io
from flask import Response, stream_with_context


# Initialize DatasetManager
dataset_model = DatasetModel()
user_model = UserModel()

# API Endpoints
@datasets_bp.route('/add_dataset', methods=['POST'])
def add_dataset():
    try:
        data = request.form
        file = request.files['file']
        user_id = data.get('user_id')
        project_name = data.get('project_name')

        if not user_id or not project_name:
            return jsonify({"error": "Missing required fields: user_id, project_name"}), 400


        dataset_id = dataset_model.create_dataset(
            user_id=data['user_id'],
            file=file,
            project_name= data['project_name'],
        )
        user_model.add_dataset(data['user_id'] ,dataset_id)
        return jsonify({"message": "Dataset created successfully", "dataset_id": dataset_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@datasets_bp.route('/get_dataset', methods=['GET'])
def get_dataset():
    dataset_id = ObjectId(request.args.get('dataset_id'))
    dataset = dataset_model.get_dataset(dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(dataset), 200



@datasets_bp.route('/get_dataset_csv', methods=['GET'])
def get_dataset_csv():
    dataset_id = request.args.get('dataset_id')
    if not dataset_id:
        return jsonify({"error": "Missing dataset_id parameter"}), 400

    dataset = dataset_model.get_dataset(dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404

    # Retrieve the file from GridFS
    grid_out = dataset_model.get_dataset_csv(dataset_id)

    # Stream file content instead of loading it fully into memory
    def generate():
        while chunk := grid_out.read(8192):  # Read in chunks of 8KB
            yield chunk

    return Response(stream_with_context(generate()), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename={dataset['filename']}.csv"
    })



@datasets_bp.route('/update_dataset', methods=['PUT'])
def update_dataset():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    dataset_id = data.get("dataset_id")
    update_fields = data.get("dataset_fields")

    if not dataset_id or not update_fields:
        return jsonify({"error": "Missing required fields: dataset_id, dataset_fields"}), 400

    try:
        dataset_id = ObjectId(dataset_id)
    except errors.InvalidId:
        return jsonify({"error": "Invalid dataset_id format"}), 400

    modified_count = dataset_model.update_dataset(dataset_id, update_fields)

    if modified_count == 0:
        return jsonify({"message": "No changes made or dataset not found."}), 404

    return jsonify({"message": "Dataset updated successfully."}), 200


@datasets_bp.route('/delete_dataset', methods=['DELETE'])
def delete_dataset():
    dataset_id = ObjectId(request.args.get('dataset_id'))
    user_id = ObjectId(request.args.get('user_id'))
    removed_dataset_from_user = user_model.remove_dataset(user_id, dataset_id)
    if removed_dataset_from_user > 0:
        deleted_count = dataset_model.delete_dataset(dataset_id)
    else:
        return jsonify({"error": "User not found or dataset not found in user's list."}), 404

    if deleted_count == 0:
        return jsonify({"error": "Dataset not found or already deleted."}), 404
    return jsonify({"message": "Dataset deleted successfully."}), 200

@datasets_bp.route('/get_dataset_column', methods=['GET'])
def get_dataset_column():
    dataset_id = request.args.get('dataset_id')
    columns = dataset_model.get_column_names(dataset_id)
    return jsonify({"columns": columns}), 200   