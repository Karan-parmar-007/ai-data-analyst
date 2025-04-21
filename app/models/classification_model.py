from bson import ObjectId
from app.utils.db import db

class ClassificationModel:
    def __init__(self):
        self.classifications_collection = db["classifications"]
        self.models_collection = db["models"]

    def add_classification(self, classification_name):
        existing = self.classifications_collection.find_one({"classification_name": classification_name})
        if existing:
            return {"error": "Classification already exists"}
        result = self.classifications_collection.insert_one({"classification_name": classification_name, "models": []})
        return {"message": "Classification added", "id": str(result.inserted_id)}

    def get_all_classifications(self):
        classifications = self.classifications_collection.find()
        return [{"_id": str(c["_id"]), "classification_name": c["classification_name"]} for c in classifications]

    def update_classification(self, classification_id, new_name):
        result = self.classifications_collection.update_one({"_id": ObjectId(classification_id)}, {"$set": {"classification_name": new_name}})
        if result.modified_count:
            return {"message": "Classification updated"}
        return {"error": "No changes made or classification not found"}

    def add_model_to_classification(self, classification_id, model_name, hyperparameters):
        classification = self.classifications_collection.find_one({"_id": ObjectId(classification_id)})
        if not classification:
            return {"error": "Classification not found"}
        
        model_data = {
            "model_name": model_name,
            "classification_id": classification_id,
            "hyperparameters": hyperparameters
        }
        result = self.models_collection.insert_one(model_data)
        self.classifications_collection.update_one(
            {"_id": ObjectId(classification_id)},
            {"$push": {"models": str(result.inserted_id)}}
        )
        return {"message": "Model added", "model_id": str(result.inserted_id)}

    def get_all_classifications_with_models(self):
        classifications = self.classifications_collection.find()
        result = {}
        for classification in classifications:
            models = self.models_collection.find({"classification_id": str(classification["_id"])})
            result[classification["classification_name"]] = {
                model["model_name"]: model["hyperparameters"] for model in models
            }
        return result

    def get_models_for_classification(self, classification_id):
        models = self.models_collection.find({"classification_id": classification_id})
        return [{"_id": str(m["_id"]), "model_name": m["model_name"]} for m in models]

    def delete_model(self, model_id):
        model = self.models_collection.find_one({"_id": ObjectId(model_id)})
        if not model:
            return {"error": "Model not found"}
        
        self.models_collection.delete_one({"_id": ObjectId(model_id)})
        self.classifications_collection.update_one(
            {"_id": ObjectId(model["classification_id"])},
            {"$pull": {"models": model_id}}
        )
        return {"message": "Model deleted"}

    def delete_classification(self, classification_id):
        classification = self.classifications_collection.find_one({"_id": ObjectId(classification_id)})
        if not classification:
            return {"error": "Classification not found"}
        
        self.models_collection.delete_many({"classification_id": classification_id})
        self.classifications_collection.delete_one({"_id": ObjectId(classification_id)})
        return {"message": "Classification and associated models deleted"}
