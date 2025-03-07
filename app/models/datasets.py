from bson import ObjectId
from app.utils.db import db
import gridfs
from io import BytesIO
import pandas as pd

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
            "type_of_analysis": "",
            "is_column_important": {},
            "fill_empty_rows_using": "",
            "remove_duplicate": True,
            "scaling_and_normalization": True,
            "test_dataset_percentage": 30,
            "increase_the_size_of_dataset": False,
            "Is_preprocessing_form_filled": False,
            "is_preprocessing_done": False,
            "start_preprocessing": False,
            "models": [],
        }

        result = self.datasets_collection.insert_one(dataset)
        return str(result.inserted_id)

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
                "dataset_description": dataset.get("dataset_description", ""),
                "is_preprocessing_done": dataset.get("is_preprocessing_done", False),
                "Is_preprocessing_form_filled": dataset.get("Is_preprocessing_form_filled", False),
                "start_preprocessing": dataset.get("start_preprocessing", False),
                "test_dataset_percentage": dataset.get("test_dataset_percentage", 0),
                "remove_duplicate": dataset.get("remove_duplicate", False),
                "scaling_and_normalization": dataset.get("scaling_and_normalization", False),
                "increase_the_size_of_dataset": dataset.get("increase_the_size_of_dataset", False)
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
            {"$set": update_fields}
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
    
    def save_chart(self, dataset_id: str, chart_data: dict) -> str:
        """Save a chart configuration to the dataset document."""
        try:
            # Generate a unique chart ID (could use ObjectId or a simpler counter)
            chart_id = str(ObjectId())
            chart_entry = {
                "chart_id": chart_id,
                "chart_type": chart_data["chart_type"],
                "x_axis": chart_data["x_axis"],
                "y_axis": chart_data["y_axis"],
                "chart_data": chart_data["chart_data"]  # The actual chart configuration
            }
            
            # Add the chart to the dataset's charts array (create array if it doesn't exist)
            result = self.datasets_collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$push": {"charts": chart_entry}},
                upsert=True
            )
            if result.modified_count == 0 and result.upserted_id is None:
                raise ValueError("Failed to save chart: dataset not found or update failed")
            return chart_id
        except Exception as e:
            print(f"Error saving chart for dataset {dataset_id}: {str(e)}")
            raise

    def get_saved_charts(self, dataset_id: str) -> list:
        """Retrieve all saved charts for a dataset."""
        try:
            dataset = self.datasets_collection.find_one({"_id": ObjectId(dataset_id)}, {"charts": 1})
            if not dataset or "charts" not in dataset:
                return []
            return dataset["charts"]
        except Exception as e:
            print(f"Error fetching charts for dataset {dataset_id}: {str(e)}")
            return []
        
    def delete_chart(self, dataset_id: str, chart_id: str) -> int:
        """Delete a specific chart from the dataset's charts array."""
        try:
            result = self.datasets_collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$pull": {"charts": {"chart_id": chart_id}}}
            )
            return result.modified_count  # Returns 1 if a chart was removed, 0 if not found
        except Exception as e:
            print(f"Error deleting chart {chart_id} from dataset {dataset_id}: {str(e)}")
            raise