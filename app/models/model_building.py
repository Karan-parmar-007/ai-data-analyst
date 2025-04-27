from bson import ObjectId
from app.utils.db import db

class ModelToBeBuilt:
    def __init__(self):
        """Initialize the model building manager with the model_building collection."""
        self.model_building_collection = db["model_building"]

    def create_model(self, dataset_id: str, model_name: str, classification_id: str, hyperparameters: dict) -> str:
        """Create a new model entry."""
        model_data = {
            "dataset_id": ObjectId(dataset_id),
            "classification_id": ObjectId(classification_id),
            "model_name": model_name,
            "hyperparameters": hyperparameters
        }
        result = self.model_building_collection.insert_one(model_data)
        return str(result.inserted_id)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model by its _id."""
        result = self.model_building_collection.delete_one({"_id": ObjectId(model_id)})
        return result.deleted_count > 0

    def get_all_models(self) -> list:
        """Retrieve all models."""
        models = list(self.model_building_collection.find())
        return [
            {
                "_id": str(model["_id"]),
                "dataset_id": str(model["dataset_id"]),
                "classification_id": str(model["classification_id"]),
                "model_name": model["model_name"],
                "hyperparameters": model["hyperparameters"]
            }
            for model in models
        ]
