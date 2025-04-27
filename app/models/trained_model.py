from bson import ObjectId
from app.utils.db import db

class TrainedModel:
    def __init__(self):
        """Initialize the trained model manager with the trained_models collection."""
        self.trained_models_collection = db["trained_models"]

    def create_model(self, dataset_id: str, model_name: str, classification_name: str, 
                    hyperparameters: dict, data_transformation_pkl_id: str, 
                    trained_model_pkl_id: str, train_time: float = 0, 
                    metrics: dict = None, progress: list = None) -> str:
        """Create a new trained model entry."""
        model_data = {
            "dataset_id": ObjectId(dataset_id),
            "model_name": model_name,
            "classification_name": classification_name,
            "hyperparameters": hyperparameters,
            "data_transformation_pkl_id": ObjectId(data_transformation_pkl_id),
            "trained_model_pkl_id": ObjectId(str(trained_model_pkl_id)),  # Ensure valid ObjectId
            "train_time": train_time,
            "metrics": metrics or {},
            "progress": progress or []
        }
        result = self.trained_models_collection.insert_one(model_data)
        return str(result.inserted_id)

    def get_model_by_id(self, model_id: str) -> dict:
        """Get a trained model by its ID."""
        model = self.trained_models_collection.find_one({"_id": ObjectId(model_id)})
        if model:
            return {
                "_id": str(model["_id"]),
                "dataset_id": str(model["dataset_id"]),
                "model_name": model["model_name"],
                "classification_name": model["classification_name"],
                "hyperparameters": model["hyperparameters"],
                "data_transformation_pkl_id": str(model["data_transformation_pkl_id"]),
                "trained_model_pkl_id": str(model["trained_model_pkl_id"]),
                "train_time": model.get("train_time", 0),
                "metrics": model.get("metrics", {}),
                "progress": model.get("progress", [])
            }
        return None

    def get_models_by_dataset(self, dataset_id: str) -> list:
        """Get all trained models for a specific dataset."""
        models = self.trained_models_collection.find({"dataset_id": ObjectId(dataset_id)})
        return [
            {
                "_id": str(model["_id"]),
                "model_name": model["model_name"],
                "classification_name": model["classification_name"],
                "hyperparameters": model["hyperparameters"],
                "data_transformation_pkl_id": str(model["data_transformation_pkl_id"]),
                "trained_model_pkl_id": str(model["trained_model_pkl_id"]),
                "train_time": model.get("train_time", 0),
                "metrics": model.get("metrics", {}),
                "progress": model.get("progress", [])
            }
            for model in models
        ]

    def delete_model(self, model_id: str) -> bool:
        """Delete a trained model by its ID."""
        result = self.trained_models_collection.delete_one({"_id": ObjectId(model_id)})
        return result.deleted_count > 0

    def delete_models_by_dataset(self, dataset_id: str) -> bool:
        """Delete all trained models for a specific dataset."""
        result = self.trained_models_collection.delete_many({"dataset_id": ObjectId(dataset_id)})
        return result.deleted_count > 0