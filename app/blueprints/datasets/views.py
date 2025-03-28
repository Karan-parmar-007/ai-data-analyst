from app.blueprints.datasets import datasets_bp
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

# Initialize DatasetManager
dataset_model = DatasetModel()
user_model = UserModel()

# API Endpoints
@datasets_bp.route("/add_dataset", methods=["POST"])
def add_dataset():
    print("Received add_dataset request")
    user_id = request.form.get("user_id")
    project_name = request.form.get("project_name")
    file = request.files.get("file")

    if not all([user_id, project_name, file]):
        print("Missing fields:", {"user_id": user_id, "project_name": project_name, "file": file})
        return jsonify({"error": "Missing required fields"}), 400

    try:
        user_id_obj = ObjectId(user_id)  # Convert to ObjectId for MongoDB
        dataset_id = dataset_model.create_dataset(user_id, file, project_name)  # Returns string
        dataset_id_obj = ObjectId(dataset_id)
        print(f"Inserted dataset: {dataset_id}")

        update_result = user_model.users_collection.update_one(
            {"_id": user_id_obj},
            {"$push": {"datasets": dataset_id_obj}}
        )
        print(f"Update result: matched={update_result.matched_count}, modified={update_result.modified_count}")

        if update_result.matched_count == 0:
            print(f"User with _id {user_id} not found")
            return jsonify({"error": "User not found"}), 404

        response = {
            "_id": dataset_id,  # String
            "filename": project_name,  # Matches your DatasetModel
            "dataset_description": "",
            "is_preprocessing_done": False,
            "Is_preprocessing_form_filled": False,
            "start_preprocessing": False,
            "test_dataset_percentage": 30,  # Default from create_dataset
            "remove_duplicate": True,  # Default from create_dataset
            "scaling_and_normalization": True,  # Default from create_dataset
            "increase_the_size_of_dataset": False,
        }
        print("Response:", response)
        return jsonify(response), 200
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error in add_dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

@datasets_bp.route('/get_column_names', methods=['GET'])
def get_column_names():
    dataset_id = request.args.get('dataset_id')
    if not dataset_id:
        return jsonify({"error": "Dataset ID required"}), 400

    dataset = dataset_model.get_dataset(dataset_id)
    if not dataset or "file_id" not in dataset:
        return jsonify({"error": "Dataset not found or missing file_id"}), 404

    try:
        column_names = dataset_model.get_column_names(dataset_id)
        print(f"Successfully fetched column names for dataset {dataset_id}: {column_names}")
        return jsonify({"column_names": column_names}), 200
    except Exception as e:
        print(f"Error fetching column names for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": f"Failed to fetch column names: {str(e)}"}), 500


@datasets_bp.route('/start_preprocessing', methods=['POST'])
def start_preprocessing():
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
    
    datasets_to_be_Preprocessed_model = DatasetsToBePreprocessedModel()
    inserted_success = datasets_to_be_Preprocessed_model.create_dataset_to_be_preprocessed(dataset_id)

    if not inserted_success:
        return jsonify({"message": "wasn't able to set prrprocessing as True."}), 404

    modified_count = dataset_model.update_dataset(dataset_id, update_fields)

    if modified_count == 0:
        return jsonify({"message": "wasn't able to set prrprocessing as True."}), 404

    return jsonify({"message": "dataset will be preprocessed."}), 200

@datasets_bp.route('/get_preprocessing_status', methods=['GET'])
def get_preprocessing_status():
    dataset_id = request.args.get('dataset_id')
    status = dataset_model.get_preprocessing_status(dataset_id)
    return jsonify({"status": status}), 200
   
    
# Dataset Saving routes currently removed (will be reactivated if needed)

@datasets_bp.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json() or {}
    dataset_id = data.get("dataset_id")
    chart_type = data.get("chart_type")
    x_axis = data.get("x_axis")
    y_axis = data.get("y_axis")

    if not all([dataset_id, chart_type, x_axis, y_axis]):
        return jsonify({"error": "Missing required fields: dataset_id, chart_type, x_axis, y_axis"}), 400

    try:
        dataset = dataset_model.get_dataset(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        if "file_id" not in dataset or not dataset["file_id"]:
            return jsonify({"error": "Dataset missing file_id"}), 400

        file_id = dataset["file_id"]
        grid_out = dataset_model.fs.get(ObjectId(file_id))
        df = pd.read_csv(BytesIO(grid_out.read()))
        print(f"Dataset columns: {df.columns.tolist()}")

        if x_axis not in df.columns or y_axis not in df.columns:
            return jsonify({"error": f"Columns {x_axis} or {y_axis} not found in dataset"}), 400

        x_data = df[x_axis].dropna().tolist()
        y_data = df[y_axis].dropna().tolist()
        print(f"x_data: {x_data}")
        print(f"y_data: {y_data}")

        if not x_data or not y_data:
            return jsonify({"error": f"No valid data in columns {x_axis} or {y_axis} after removing NaN"}), 400

        if chart_type == "scatter":
            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]

        return jsonify({"x_data": x_data, "y_data": y_data}), 200
    except Exception as e:
        print(f"Error generating visualization for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

    
    
    
    



