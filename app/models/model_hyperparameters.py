from bson import ObjectId
from app.utils.db import db

class ModelHyperparameters:
    def __init__(self):
        self.models_collection = db["models"]

    def get_hyperparameters_for_model(self, model_id):
        model = self.models_collection.find_one({"_id": ObjectId(model_id)})
        if not model:
            return {"error": "Model not found"}
        return {"model_name": model["model_name"], "hyperparameters": model["hyperparameters"]}

    def update_model_hyperparameters(self, model_id, updated_hyperparameters):
        result = self.models_collection.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"hyperparameters": updated_hyperparameters}}
        )
        if result.modified_count:
            return {"message": "Hyperparameters updated"}
        return {"error": "No changes made or model not found"}
