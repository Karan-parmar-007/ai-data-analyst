from bson import ObjectId
from app.utils.db import db
import gridfs
import pandas as pd
import pickle
import io
from io import BytesIO

class DatasetModel:
    def __init__(self):
        """Initialize the dataset manager with the datasets collection and GridFS."""
        self.datasets_collection = db["datasets"]
        self.fs = gridfs.GridFS(db)

    def create_dataset(self, user_id: str, file, project_name: str) -> str:
        """Create a new dataset entry by storing the CSV file in GridFS."""
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are allowed.")

        # Store the file in GridFS
        file_id = self.fs.put(file, filename=file.filename)

        dataset = {
            "user_id": ObjectId(user_id),
            "file_id": file_id,
            "filename": project_name,
            "dataset_description": "",
            "datatype_of_each_column": {},
            "usage_of_each_column": {},
            "fill_empty_rows_using": "",
            "remove_duplicate": True,
            "standardization_necessary": True,
            "normalization_necessary": False,
            "test_dataset_percentage": 30,
            "increase_the_size_of_dataset": False,
            "Is_preprocessing_form_filled": False,
            "is_preprocessing_done": False,
            "start_preprocessing": False,
            "models": [],
            "fill_string_type_columns": False,
            "dimensionality_reduction": False,
            "target_column": "",
            "remove_highly_correlated_columns": False,
            "data_transformation_file_and_transformaed_dataset": {},
            "what_user_wants_to_do": ""
        }

        result = self.datasets_collection.insert_one(dataset)
        return str(result.inserted_id)

    def store_artifacts(self, dataset_id: str, config_pickle_obj):
        """
        Stores the data transformation configuration (as a pickle) in GridFS,
        then updates the dataset document with the artifact file ID nested under
        the `data_transformation_file_and_transformaed_dataset` field.
        """
        try:
            # Serialize the configuration object using pickle.
            config_bytes = pickle.dumps(config_pickle_obj)
            config_file_id = self.fs.put(config_bytes, filename="data_transformation_config.pkl")

            # Use MongoDB aggregation pipeline to handle null parent field
            self.datasets_collection.update_one(
                {"_id": ObjectId(dataset_id)},
                [
                    # Ensure the parent field is a document (not null)
                    {
                        "$set": {
                            "data_transformation_file_and_transformaed_dataset": {
                                "$ifNull": ["$data_transformation_file_and_transformaed_dataset", {}]
                            }
                        }
                    },
                    # Set the nested config_file_id
                    {
                        "$set": {
                            "data_transformation_file_and_transformaed_dataset.config_file_id": config_file_id
                        }
                    }
                ]
            )

            return True

        except Exception as e:
            print(f"Error storing artifacts: {e}")
            raise e

    def store_model_artifact(self, dataset_id: str, model_artifact: dict):
        """
        Stores a model training artifact in GridFS and updates the dataset's 'models' array.
        The model_artifact dictionary should contain:
          - model_name: str, e.g., "LogisticRegression"
          - train_time: float, training time in seconds
          - metrics: dict, e.g., {'train_accuracy': ..., 'test_accuracy': ..., ...}
          - artifact_store_response: (optional) response from any previous storage steps
          - trained_model: the actual scikit-learn model object (this will be pickled and stored)
        
        After pickling the trained model, only its file ID is stored in the document.
        """
        try:
            # Extract and pickle the trained model object.
            model_obj = model_artifact.get("trained_model")
            if model_obj is None:
                raise ValueError("The model_artifact dictionary must include a 'trained_model' key.")

            # Remove the model object from the dictionary so it doesn't get stored directly.
            del model_artifact["trained_model"]

            # Serialize the model object using pickle.
            model_pickle_bytes = pickle.dumps(model_obj)
            # Store the pickled model in GridFS.
            model_file_id = self.fs.put(model_pickle_bytes, filename=f"{model_artifact.get('model_name')}_model.pkl")
            
            # Add the GridFS file ID to the artifact dictionary.
            model_artifact["model_file_id"] = model_file_id

            # Append the artifact details to the 'models' array in the dataset document.
            self.datasets_collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$push": {"models": model_artifact}}
            )
            return True

        except Exception as e:
            print(f"Error storing model artifact: {e}")
            raise e

    def get_dataset(self, dataset_id: str) -> dict:
        try:
            dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
            if not dataset:
                return {}
            return {
                "_id": str(dataset["_id"]),
                "user_id": str(dataset["user_id"]),
                "file_id": str(dataset["file_id"]),  # This line assumes file_id exists
                "filename": dataset["filename"],
                "data_transformation_file_and_transformaed_dataset": dataset.get("data_transformation_file_and_transformaed_dataset", {}),
                "datatype_of_each_column": dataset.get("datatype_of_each_column", {}),
                "usage_of_each_column": dataset.get("usage_of_each_column", {}),
                "dataset_description": dataset.get("dataset_description", ""),
                "is_preprocessing_done": dataset.get("is_preprocessing_done", False),
                "Is_preprocessing_form_filled": dataset.get("Is_preprocessing_form_filled", False),
                "start_preprocessing": dataset.get("start_preprocessing", False),
                "test_dataset_percentage": dataset.get("test_dataset_percentage", 0),
                "remove_duplicate": dataset.get("remove_duplicate", False),
                "standardization_necessary": dataset.get("standardization_necessary", False),
                "normalization_necessary": dataset.get("normalization_necessary", False),
                "fill_empty_rows_using": dataset.get("fill_empty_rows_using", ""),
                "increase_the_size_of_dataset": dataset.get("increase_the_size_of_dataset", False),
                "dimensionality_reduction": dataset.get("dimensionality_reduction", False),
                "standardization_necessary": dataset.get("standardization_necessary", False),
                "normalization_necessary": dataset.get("normalization_necessary", False),
                "target_column": dataset.get("target_column", ""),
                "remove_highly_correlated_columns": dataset.get("remove_highly_correlated_columns", False),
                "what_user_wants_to_do": dataset.get("what_user_wants_to_do", ""),
            }
        except Exception as e:
            print(f"Error fetching dataset {dataset_id}: {str(e)}")
            return {}
    
    def get_dataset_csv(self, dataset_id):
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        dataset["file_id"] = str(dataset["file_id"])
        grid_out = self.fs.get(ObjectId(dataset["file_id"]))
        return grid_out



    def update_dataset(self, dataset_id: str, update_fields: dict) -> int:
        """Update specific fields of a dataset."""
        result = self.datasets_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": update_fields},  # Use $set to update fields
        )
        return result.modified_count

    def delete_dataset(self, dataset_id: str) -> int:
        """Delete a dataset and its file from GridFS."""
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            return 1

        # Delete the file from GridFS
        file_id = dataset.get("file_id")
        if file_id:
            self.fs.delete(ObjectId(file_id))

        # Delete the dataset document
        result = self.datasets_collection.delete_one({"_id": ObjectId(dataset_id)})
        return result.deleted_count

    def delete_datasets_by_user_id(self, user_id: str) -> int:
        """
        Delete all datasets associated with a given user_id and remove the corresponding files from GridFS.

        Args:
            user_id (str): The ID of the user whose datasets are to be deleted.

        Returns:
            int: The number of deleted datasets.
        """
        # Convert the user_id to ObjectId for querying
        user_object_id = ObjectId(user_id)

        # Find all datasets with the given user_id
        datasets = list(self.datasets_collection.find({"user_id": user_object_id}))

        if not datasets:
            return 1

        # Delete files from GridFS associated with the datasets
        for dataset in datasets:
            file_id = dataset.get("file_id")
            if file_id:
                try:
                    self.fs.delete(file_id)
                except Exception as e:
                    print(f"Warning: Failed to delete file with ID {file_id}: {e}")

        # Delete datasets from the collection
        delete_result = self.datasets_collection.delete_many({"user_id": user_object_id})

        return delete_result.deleted_count

    def get_dataset_filename(self, dataset_id: str) -> str:
        """Fetch the dataset name based on dataset ID."""
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)}, {"filename": 1})
        return dataset["filename"] if dataset else "Unknown Dataset"
    
    def get_column_names(self, dataset_id: str) -> list:
        """Retrieve all column names from a dataset stored in GridFS."""
        try:
            dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
            if not dataset:
                raise ValueError("Dataset not found")

            file_id = dataset.get("file_id")
            if not file_id:
                raise ValueError("Dataset missing file_id")

            try:
                grid_out = self.fs.get(ObjectId(file_id))
                file_data = grid_out.read()
                print(f"Successfully fetched file data for file_id {file_id}, length: {len(file_data)}")
            except Exception as e:
                raise ValueError(f"Failed to retrieve file from GridFS with file_id {file_id}: {str(e)}")

            try:
                df = pd.read_csv(BytesIO(file_data))
                return df.columns.tolist()
            except Exception as e:
                raise ValueError(f"Failed to parse CSV data: {str(e)}")
        except ValueError as e:
            raise e  # Re-raise for specific error handling in route
        except Exception as e:
            print(f"Unexpected error in get_column_names for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_preprocessing_status(self, dataset_id: str) -> bool:
        """Check if the preprocessing form is filled for a dataset."""
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        return dataset.get("is_preprocessing_done", False)
    

    def start_preprocessing(self, dataset_id: str) -> int:
        """
        Set start_preprocessing to True for a given dataset.

        Args:
            dataset_id (str): The ID of the dataset to update.

        Returns:
            int: 1 if updated, 0 if already True or dataset not found.
        """
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)}, {"start_preprocessing": 1})

        if dataset and dataset.get("start_preprocessing") is True:
            return 0  # Already True, no update needed

        result = self.datasets_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {"start_preprocessing": True}}
        )

        return result.modified_count  # 1 if updated, 0 if dataset not found



    def stop_preprocessing(self, dataset_id: str) -> int:
        """
        Set start_preprocessing to False for a given dataset.

        Args:
            dataset_id (str): The ID of the dataset to update.

        Returns:
            int: 1 if updated, 0 if already False or dataset not found.
        """
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)}, {"start_preprocessing": 1})

        if dataset and dataset.get("start_preprocessing") is False:
            return 0  # Already False, no update needed

        result = self.datasets_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {"start_preprocessing": False}}
        )

        return result.modified_count  # 1 if updated, 0 if dataset not found


    def update_preprocessing_status(self, dataset_id: str, is_preprocessing_done: bool) -> int:
        """
        Update the preprocessing status of a dataset.

        Args:
            dataset_id (str): The ID of the dataset to update.
            is_preprocessing_done (bool): The new status to set.

        Returns:
            int: 1 if updated, 0 if already set or dataset not found.
        """
        dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)}, {"is_preprocessing_done": 1})

        if dataset and dataset.get("is_preprocessing_done") == is_preprocessing_done:
            return 0  # Already set, no update needed

        result = self.datasets_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {"is_preprocessing_done": is_preprocessing_done}}
        )

        return result.modified_count  # 1 if updated, 0 if dataset not found

    # In app/models/datasets.py
    def update_dataset_file(self, dataset_id: str, new_df: pd.DataFrame, is_preprocessing_done: bool) -> None:
        """Update the dataset file with the preprocessed DataFrame and set is_preprocessing_done."""
        from io import BytesIO
        from bson import ObjectId
        
        try:
            # Convert DataFrame to CSV in memory
            csv_buffer = BytesIO()
            new_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Upload new file to GridFS
            new_file_id = self.fs.put(csv_buffer, filename=f"{dataset_id}_preprocessed.csv")
            
            # Get the current dataset to retrieve the old file_id
            dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)})
            if not dataset:
                raise ValueError("Dataset not found.")
            
            old_file_id = dataset.get("file_id")
            
            # Update the dataset document with new file_id and is_preprocessing_done
            update_fields = {
                "file_id": new_file_id,
                "is_preprocessing_done": is_preprocessing_done
            }
            result = self.datasets_collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$set": update_fields}
            )
            if result.modified_count == 0:
                print(f"Warning: No documents updated for dataset_id {dataset_id}")  # Debug
            
            # Delete the old file from GridFS if it exists
            if old_file_id:
                self.fs.delete(old_file_id)
        except Exception as e:
            print(f"Error in update_dataset_file for {dataset_id}: {str(e)}")  # Debug
            raise  # Re-raise to catch in cron job