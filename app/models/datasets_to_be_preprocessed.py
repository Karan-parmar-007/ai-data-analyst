from bson import ObjectId
from app.utils.db import db

class DatasetsToBePreprocessedModel:
    def __init__(self):
        """Initialize the datasets to be preprocessed manager with the datasets_to_be_preprocessed collection."""
        self.datasets_to_be_preprocessed_collection = db["datasets_to_be_preprocessed"]

    def create_dataset_to_be_preprocessed(self, dataset_id: str) -> str:
        """Create a new dataset to be preprocessed entry."""
        dataset_to_be_preprocessed = {
            "dataset_id": ObjectId(dataset_id),
        }

        result = self.datasets_to_be_preprocessed_collection.insert_one(dataset_to_be_preprocessed)
        return str(result.inserted_id)

