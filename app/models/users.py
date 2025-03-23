import datetime
from bson import ObjectId
from flask import request, jsonify
from app.utils.db import db

class UserModel:
    def __init__(self):
        """Initialize the user manager with the users collection."""
        self.users_collection = db["users"]

    # def create_user(self, name: str, email: str) -> str:
    #     """Create a new user with an empty datasets list."""
    #     user = {"name": name, "email": email, "datasets": []}
    #     result = self.users_collection.insert_one(user)
    #     return str(result.inserted_id)

    def create_user(self, clerk_user_id: str, email: str, name: str = None) -> str:
        """
        Create a new user with Clerk authentication details.
        
        Args:
            clerk_user_id: The user ID from Clerk authentication
            email: User's email from Clerk
            name: Optional user's name
        """

        # Create new user document
        user = {
            "clerk_user_id": clerk_user_id,
            "email": email,
            "name": name,
            "datasets": [],
            # "created_at": datetime.UTC(),
            # "last_login": datetime.UTC()
        }
        
        result = self.users_collection.insert_one(user)
        return str(result.inserted_id)

    def user_exists(self, clerk_user_id: str = None, email: str = None) -> bool:
        """
        Check if a user exists based on Clerk user ID or email.

        Args:
            clerk_user_id: The user ID from Clerk authentication (optional).
            email: The user's email (optional).

        Returns:
            bool: True if the user exists, False otherwise.
        """
        query = {}
        if clerk_user_id:
            query["clerk_user_id"] = clerk_user_id
        if email:
            query["email"] = email

        return self.users_collection.find_one(query) is not None


    def get_user_by_clerk_id(self, clerk_user_id: str) -> dict:
        """Fetch user details using Clerk user ID."""
        user = self.users_collection.find_one({"clerk_user_id": clerk_user_id})
        if not user:
            return {}
            
        return {
            "user_id": str(user["_id"]),
            "clerk_user_id": user["clerk_user_id"],
            "email": user["email"],
            "name": user.get("name"),
            "dataset_ids": [str(dataset_id) for dataset_id in user.get("datasets", [])]
        }
    
    def get_user_by_email(self, email: str) -> dict:
        """Fetch user details using Clerk user ID."""
        user = self.users_collection.find_one({"email": email})
        if not user:
            return {}
            
        return {
            "user_id": str(user["_id"]),
            "clerk_user_id": user["clerk_user_id"],
            "email": user["email"],
            "name": user.get("name"),
            "dataset_ids": [str(dataset_id) for dataset_id in user.get("datasets", [])]
        }

    def update_user_name(self, user_id: str, new_name: str) -> int:
        """Update the name of the user."""
        result = self.users_collection.update_one(
            {"_id": ObjectId(user_id)}, 
            {"$set": {"name": new_name}}
        )
        return result.modified_count
    
    def update_email(self, user_id: str, new_email: str) -> int:
        result = self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"email": new_email}}
        )
        return result.modified_count

    def add_dataset(self, user_id: str, dataset_id: str) -> int:
        """Add a new dataset to the user's datasets list."""
        result = self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$addToSet": {"datasets": ObjectId(dataset_id)}}  # Avoids duplicates
        )
        return result.modified_count

    def remove_dataset(self, user_id: str, dataset_id: str) -> int:
        """Remove a dataset from the user's datasets list."""
        result = self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"datasets": ObjectId(dataset_id)}}
        )
        return result.modified_count

    def delete_user(self, user_id: str) -> int:
        """Delete a user account from the collection."""
        result = self.users_collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count
    
    def get_user_details(self, email: str) -> dict:
        """Fetch user details along with dataset IDs."""
        user = self.users_collection.find_one({"email": email}, {"name": 1, "_id": 1, "datasets": 1})

        if not user:
            return {}

        return {
            "user_id": str(user["_id"]),
            "email": email,
            "name": user["name"],
            "dataset_ids": [str(dataset_id) for dataset_id in user.get("datasets", [])]
        }

# Example usage (for testing purposes):
# user_manager = UserManager()
# user_id = user_manager.create_user("Alice")
# user_manager.update_user_name(user_id, "Alicia")
# user_manager.add_dataset(user_id, "60f7f3d5368a2f1a4c8f9a3b")
# user_manager.remove_dataset(user_id, "60f7f3d5368a2f1a4c8f9a3b")
# user_manager.delete_user(user_id)
