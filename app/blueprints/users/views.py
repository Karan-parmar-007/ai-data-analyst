from app.blueprints.users import users_bp
from flask import Flask, request, jsonify
from app.models.users import UserModel
from app.models.datasets import DatasetModel
from bson.errors import InvalidId

user_model = UserModel()
dataset_model = DatasetModel()

@users_bp.route("/create-user", methods=["POST"])
def create_user():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email:
        return jsonify({"error": "Email is required."}), 400

    result = user_model.create_user(name, email)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify({"message": "User created successfully.", "user_id": result["user_id"]}), 201

@users_bp.route("/update-user-name", methods=["PUT"])
def update_user_name():
    data = request.get_json()
    new_name = data.get("name")
    user_id = data.get("user_id")
    if not new_name:
        return jsonify({"error": "New name is required."}), 400

    try:
        updated_count = user_model.update_user_name(user_id, new_name)
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400

    if updated_count:
        return jsonify({"message": "User name updated successfully."}), 200
    return jsonify({"error": "User not found."}), 404

@users_bp.route("/update_user_email", methods=["PUT"])
def update_user_email():
    data = request.get_json()
    new_email = data.get("email")
    user_id = data.get("user_id")
    if not new_email:
        return jsonify({"error": "New email is required."}), 400

    try:
        updated_count = user_model.update_email(user_id, new_email)
    except InvalidId:
        return jsonify({"error": "Invalid user ID."}), 400

    if updated_count:
        return jsonify({"message": "User email updated successfully."}), 200
    return jsonify({"error": "User not found."}), 404




@users_bp.route("/delete_user", methods=["DELETE"])
def delete_user():
    user_id = request.args.get("user_id")
    deleted_dataset = dataset_model.delete_datasets_by_user_id(user_id)
    if deleted_dataset > 0:
        try:
            deleted_count = user_model.delete_user(user_id)
        except InvalidId:
            return jsonify({"error": "Invalid user ID."}), 400

        if deleted_count:
            return jsonify({"message": "User deleted successfully."}), 200
        return jsonify({"error": "User not found."}), 404
    return jsonify({"error": "There is some error in the delete_datasets_by_user_id function"}), 404


@users_bp.route('/get_user_details', methods=['GET'])
def get_user_details():
    email = request.args.get('email')
    
    # Fetch user details (only dataset IDs)
    user_details = user_model.get_user_details(email)

    if not user_details:
        return jsonify({"error": "User not found"}), 404

    # Fetch dataset names for each dataset ID
    datasets = {}
    for dataset_id in user_details["dataset_ids"]:
        dataset_name = dataset_model.get_dataset_filename(dataset_id)
        datasets[dataset_name] = dataset_id  # Store {dataset_name: dataset_id}

    # Build final response
    response_data = {
        "user_id": user_details["user_id"],
        "email": user_details["email"],
        "name": user_details["name"],
        "datasets": datasets
    }

    return jsonify(response_data), 200









