from app.blueprints.users import users_bp
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from app.models.users import UserModel
from app.models.datasets import DatasetModel
from bson.errors import InvalidId
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

user_model = UserModel()
dataset_model = DatasetModel()

# Apply CORS to all routes (adjust origins as needed)
@users_bp.route("/create-user", methods=["POST"])
@cross_origin(supports_credentials=True)
def create_user():
    data = request.get_json() or {}
    clerk_user_id = data.get("userId")
    email = data.get("email")
    name = data.get("name")

    if not clerk_user_id or not email:
        return jsonify({"error": "Clerk user ID and email are required"}), 400

    try:
        user_id = user_model.create_user(clerk_user_id, email, name)
        return jsonify({"message": "User created", "user_id": user_id}), 201
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return jsonify({"error": "Failed to create user"}), 500

@users_bp.route("/update-user-name", methods=["PUT"])
@cross_origin(supports_credentials=True)
def update_user_name():
    data = request.get_json() or {}
    clerk_user_id = data.get("userId")  # Use Clerk ID for consistency
    new_name = data.get("name")

    if not clerk_user_id or not new_name:
        return jsonify({"error": "Clerk user ID and new name are required"}), 400

    try:
        user = user_model.get_user_by_clerk_id(clerk_user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        updated_count = user_model.update_user_name(user["user_id"], new_name)
        return jsonify({"message": "Name updated"}) if updated_count else jsonify({"error": "Update failed"}), 404
    except InvalidId as e:
        return jsonify({"error": f"Invalid ID: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error updating name: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@users_bp.route("/update_user_email", methods=["PUT"])
@cross_origin(supports_credentials=True)
def update_user_email():
    data = request.get_json() or {}
    clerk_user_id = data.get("userId")
    new_email = data.get("email")

    if not clerk_user_id or not new_email:
        return jsonify({"error": "Clerk user ID and new email are required"}), 400

    try:
        user = user_model.get_user_by_clerk_id(clerk_user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        updated_count = user_model.update_email(user["user_id"], new_email)
        return jsonify({"message": "Email updated"}) if updated_count else jsonify({"error": "Update failed"}), 404
    except InvalidId as e:
        return jsonify({"error": f"Invalid ID: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error updating email: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@users_bp.route("/delete_user", methods=["DELETE"])
@cross_origin(supports_credentials=True)
def delete_user():
    clerk_user_id = request.args.get("userId")
    if not clerk_user_id:
        return jsonify({"error": "Clerk user ID is required"}), 400

    try:
        user = user_model.get_user_by_clerk_id(clerk_user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        user_id = user["user_id"]
        datasets_deleted = False
        try:
            deleted_datasets = dataset_model.delete_datasets_by_user_id(user_id)
            datasets_deleted = True
        except Exception as e:
            logger.error(f"Failed to delete datasets for user {user_id}: {str(e)}")
            deleted_datasets = []

        deleted_count = user_model.delete_user(user_id)
        if not deleted_count:
            logger.error(f"Failed to delete user {user_id}")
            return jsonify({"error": "Failed to delete user"}), 500

        return jsonify({
            "message": "User and datasets deleted",
            "datasets_deleted": deleted_datasets,
            "datasets_status": "success" if datasets_deleted else "failed",
            "user_id": user_id
        }), 200
    except InvalidId as e:
        return jsonify({"error": f"Invalid ID: {str(e)}"}), 400
    except Exception as e:
        logger.exception(f"Error in delete_user: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@users_bp.route('/get_user_details', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_user_details():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    try:
        user_details = user_model.get_user_details(email)
        if not user_details:
            return jsonify({"error": "User not found"}), 404

        datasets = {}
        for dataset_id in user_details.get("dataset_ids", []):
            dataset_name = dataset_model.get_dataset_filename(dataset_id) or "Unknown"
            datasets[dataset_name] = dataset_id

        return jsonify({
            "user_id": user_details["user_id"],
            "email": user_details["email"],
            "name": user_details.get("name"),
            "datasets": datasets
        }), 200
    except Exception as e:
        logger.error(f"Error fetching user details: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@users_bp.route("/get-user", methods=["GET"])
@cross_origin(supports_credentials=True)
def get_user():
    clerk_user_id = request.args.get("userId")
    if not clerk_user_id:
        return jsonify({"error": "Clerk user ID is required"}), 400

    try:
        user_details = user_model.get_user_by_clerk_id(clerk_user_id)
        if not user_details:
            return jsonify({"error": "User not found"}), 404
        return jsonify(user_details), 200
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500