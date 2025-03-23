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

    def get_unprocessed_datasets(self) -> list:
        """Retrieve all datasets that have not been preprocessed yet."""
        datasets_to_be_preprocessed = list(self.datasets_to_be_preprocessed_collection.find())
        return [str(dataset["dataset_id"]) for dataset in datasets_to_be_preprocessed]
    
    def delete_dataset_to_be_preprocessed(self, dataset_id: str):
        """Delete a dataset to be preprocessed entry."""
        self.datasets_to_be_preprocessed_collection.delete_one({"dataset_id": ObjectId(dataset_id)})
        return True