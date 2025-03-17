from app.models.datasets import DatasetModel
import pandas as pd
import io
from pandas.api.types import is_numeric_dtype, is_string_dtype
from app.utils.gemini import GeminiFunctions
import numpy as np


class DatasetPreprocessing:
    def __init__(self, dataset_id: str, omit_columns: list = None):
        """Initialize with dataset_id and optional columns to omit."""
        self.dataset_id = dataset_id
        self.omit_columns = omit_columns if omit_columns is not None else []  # Default to empty list
        self.gemini_functions = GeminiFunctions()
        self.dataset_model = DatasetModel()

    def gridout_to_dataframe(self, grid_out):
        """Convert GridFS file to a pandas DataFrame."""
        try:
            data_bytes = grid_out.read()
            data_io = io.BytesIO(data_bytes)
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            print(f"Error while converting grid_out to DataFrame: {e}")
            return None

    def check_and_convert_datatypes(self, df, datatype_dict):
        """Check and convert DataFrame column types, omitting non-float columns."""
        try:
            # Filter out columns to omit (user-specified or non-float)
            columns_to_keep = [
                col for col in df.columns 
                if col not in self.omit_columns and (col not in datatype_dict or datatype_dict[col] == "float")
            ]
            df = df[columns_to_keep].copy()

            # Process remaining columns
            for column in df.columns:
                expected_type = datatype_dict.get(column, "float")  # Default to float if not specified
                original_empty_mask = df[column].isna() | (df[column].astype(str).str.strip() == '')

                if not self.is_same_datatype(df[column], expected_type):
                    df[column] = self.convert_to_float(df[column])

                # Drop rows where conversion failed (and wasn’t originally empty)
                conversion_failed_mask = df[column].isna() & ~original_empty_mask
                df = df[~conversion_failed_mask].copy()

            # Ensure all columns are float
            df = self.ensure_float(df)
            return df
        except Exception as e:
            print(f"Error in checking and converting datatypes: {e}")
            return None

    def is_same_datatype(self, series, expected_type):
        """Check if series matches the expected data type (float only)."""
        non_null_series = series[series.notnull()]
        return expected_type == "float" and pd.api.types.is_float_dtype(non_null_series)

    def convert_to_float(self, series):
        """Convert series to float type."""
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            print(f"Error converting {series.name} to float: {e}")
            return series

    def ensure_float(self, df):
        """Ensure all columns are float for ML compatibility."""
        for column in df.columns:
            if not pd.api.types.is_float_dtype(df[column]):
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
        return df

    def main(self):
        """Main method to preprocess the dataset."""
        dataset_details = self.dataset_model.get_dataset(self.dataset_id)
        if not dataset_details:
            print(f"Dataset with ID {self.dataset_id} not found.")
            return

        # Load dataset from GridFS
        grid_out = self.dataset_model.get_dataset_csv(self.dataset_id)
        df = self.gridout_to_dataframe(grid_out)
        if df is None:
            print("Failed to load dataset into DataFrame.")
            return

        # Define expected datatypes based on your dataset
        expected_datatypes = {
            "AMR_female(per_1000_female_adults)": "float",
            "AMR_male(per_1000_male_adults)": "float",
            "Average_CDR": "float",
            "Average_GDP(M$)": "float",
            "Average_GDP_per_capita($)": "float",
            "Average_HEXP($)": "float",
            "Average_Pop(thousands people)": "float",
            "Continent": "category",
            "Countries": "category",
            "Development_level": "category"
        }

        # Preprocess dataset
        df = self.check_and_convert_datatypes(df, expected_datatypes)
        if df is None:
            print("Error in converting dataset, returned empty df.")
            return

        # Save preprocessed dataset
        self.save_preprocessed_dataset(df)

    def save_preprocessed_dataset(self, df):
        """Save the preprocessed DataFrame back to GridFS."""
        try:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            file_id = self.dataset_model.fs.put(buffer.getvalue().encode('utf-8'), filename=f"{self.dataset_id}_preprocessed.csv")
            self.dataset_model.update_dataset(self.dataset_id, {"preprocessed_file_id": file_id, "is_preprocessing_done": True})
            print(f"Preprocessed dataset saved for ID {self.dataset_id}")
        except Exception as e:
            print(f"Error saving preprocessed dataset: {e}")