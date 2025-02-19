from bson import ObjectId
from app.utils.db import db

class UserModel:
    def __init__(self):
        """Initialize the user manager with the users collection."""
        self.users_collection = db["users"]

    def create_user(self, name: str, email: str) -> str:
        """Create a new user with an empty datasets list."""
        user = {"name": name, "email": email, "datasets": []}
        result = self.users_collection.insert_one(user)
        return str(result.inserted_id)

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
