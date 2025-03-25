from app.models.datasets import DatasetModel
import pandas as pd
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoveDuplicates:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def gridout_to_dataframe(self, grid_out):
        try:
            data_bytes = grid_out.read()
            data_io = io.BytesIO(data_bytes)
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            logger.error(f"Error converting grid_out to DataFrame: {e}")
            return None
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with potential duplicate rows.
        
        Returns:
            pd.DataFrame: DataFrame with duplicate rows removed.
        """
        original_rows = df.shape[0]
        df = df.drop_duplicates()  # Returns a new DataFrame with duplicates removed
        rows_after_removal = df.shape[0]
        logger.info(f"Removed {original_rows - rows_after_removal} duplicate rows from dataset {self.dataset_id}.")
        return df

    def main(self):
        dataset_model = DatasetModel()
        dataset_details = dataset_model.get_dataset(self.dataset_id)
        if not dataset_details:
            logger.error(f"Dataset with ID {self.dataset_id} not found.")
            return False  # Explicit failure return
        
        grid_out = dataset_model.get_dataset_csv(self.dataset_id)
        df = self.gridout_to_dataframe(grid_out)
        if df is None or df.empty:
            logger.error(f"Failed to convert dataset {self.dataset_id} to DataFrame or DataFrame is empty.")
            return False  # Explicit failure return
        
        df = self.remove_duplicates(df)
        print(df)
        dataset_model.update_dataset_file(self.dataset_id, df, is_preprocessing_done=False)
        return True