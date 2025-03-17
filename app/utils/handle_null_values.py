from app.models.datasets import DatasetModel
import pandas as pd
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandleNullValues:
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

    def handle_null_values(self, df: pd.DataFrame, expected_datatypes: dict, fill_empty_rows_using: str, fill_string_type_columns: bool) -> pd.DataFrame:
        """
        Handle null values in a DataFrame based on column types and specified rules.
        
        Args:
            df (pd.DataFrame): Input DataFrame with potential null values.
            expected_datatypes (dict): Dictionary mapping column names to their expected data types.
            fill_empty_rows_using (str): Method to fill nulls in numerical columns ("mean", "median", "mode", "custom_value").
            fill_string_type_columns (bool): Whether to fill string columns with the most frequent value or drop rows.
        
        Returns:
            pd.DataFrame: DataFrame with null values handled according to the rules.
        """
        if not expected_datatypes:
            logger.info(f"No expected datatypes specified for dataset {self.dataset_id}. Skipping null value handling.")
            return df  # Return unchanged if no types specified

        rows_to_drop = set()
        
        for column in df.columns:
            expected_type = expected_datatypes.get(column)
            if expected_type is None:
                continue  # Skip columns not in expected_datatypes
            
            if expected_type == "str":
                if fill_string_type_columns:
                    mode = df[column].mode()
                    if not mode.empty and len(mode) == 1:
                        df[column] = df[column].fillna(mode.iloc[0])
                    else:
                        null_indices = df[df[column].isnull()].index
                        rows_to_drop.update(null_indices)
                else:
                    null_indices = df[df[column].isnull()].index
                    rows_to_drop.update(null_indices)
            
            elif expected_type == "category":
                mode = df[column].mode()
                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])
            
            elif expected_type in ["int", "float"]:
                if fill_empty_rows_using == "mean":
                    fill_value = df[column].mean()
                elif fill_empty_rows_using == "median":
                    fill_value = df[column].median()
                elif fill_empty_rows_using == "mode":
                    mode = df[column].mode()
                    fill_value = mode.iloc[0] if not mode.empty else np.nan
                elif fill_empty_rows_using == "custom_value":
                    fill_value = 0  # Adjust as needed
                else:
                    logger.error(f"Unknown fill_empty_rows_using method: {fill_empty_rows_using}")
                    fill_value = np.nan
                
                df[column] = df[column].fillna(fill_value)
                if expected_type == "int":
                    df[column] = df[column].round().astype("Int64")
            
            elif expected_type in ["date_time", "time", "timedelta"]:
                mode = df[column].mode()
                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])
        
        if rows_to_drop:
            df = df.drop(index=rows_to_drop)
        
        logger.info(f"Null values handled for dataset {self.dataset_id}")
        return df

    def main(self):
        dataset_model = DatasetModel()
        dataset_details = dataset_model.get_dataset(self.dataset_id)
        if not dataset_details:
            logger.error(f"Dataset with ID {self.dataset_id} not found.")
            return False
        
        grid_out = dataset_model.get_dataset_csv(self.dataset_id)
        df = self.gridout_to_dataframe(grid_out)
        if df is None or df.empty:
            logger.error(f"Failed to convert dataset {self.dataset_id} to DataFrame or DataFrame is empty.")
            return False
        
        fill_empty_rows_using = dataset_details.get("fill_empty_rows_using", "")
        fill_string_type_columns = dataset_details.get("fill_string_type_columns", False)
        if not fill_empty_rows_using:
            logger.info(f"Fill empty rows using method not specified for dataset {self.dataset_id}. Defaulting to 'mean'.")
            fill_empty_rows_using = "mean"  # Default to mean
        
        expected_datatypes = dataset_details.get("datatype_of_each_column", {})
        df = self.handle_null_values(df, expected_datatypes, fill_empty_rows_using, fill_string_type_columns)
        dataset_model.update_dataset_file(self.dataset_id, df, is_preprocessing_done=False)
        logger.info(f"Dataset {self.dataset_id} null handling completed successfully")
        return True