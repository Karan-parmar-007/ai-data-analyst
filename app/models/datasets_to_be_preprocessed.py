from bson import ObjectId
from app.utils.db import db

class DatasetsToBePreprocessedModel:
    def __init__(self):
        """Initialize the datasets to be preprocessed manager with the datasets collection."""
        self.datasets_to_be_preprocessed_collection = db["datasets"]  # Changed to use 'datasets' collection

    def create_dataset_to_be_preprocessed(self, dataset_id: str) -> str:
        """Mark an existing dataset as needing preprocessing by ensuring is_preprocessing_done is False."""
        # Instead of creating a new entry, update the existing dataset to ensure preprocessing is pending
        result = self.datasets_to_be_preprocessed_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {"is_preprocessing_done": False}},  # Ensure preprocessing flag is False
            upsert=False  # Don't create a new dataset, assume it exists
        )
        if result.matched_count == 0:
            raise ValueError(f"Dataset {dataset_id} not found")
        return dataset_id  # Return the dataset_id instead of a new inserted_id

    def get_unprocessed_datasets(self) -> list:
        """Retrieve all datasets that have not been preprocessed yet."""
        datasets_to_be_preprocessed = list(self.datasets_to_be_preprocessed_collection.find())
        return [str(dataset["dataset_id"]) for dataset in datasets_to_be_preprocessed]
    
    def delete_dataset_to_be_preprocessed(self, dataset_id: str):
        """Delete a dataset to be preprocessed entry."""
        self.datasets_to_be_preprocessed_collection.delete_one({"dataset_id": ObjectId(dataset_id)})
        return True