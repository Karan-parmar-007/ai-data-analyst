from app.models.datasets import DatasetModel
import pandas as pd
import io
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype
from app.utils.gemini import GeminiFunctions
import numpy as np


class DatasetPreprocessing:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.gemini_functions = GeminiFunctions()

    def gridout_to_dataframe(self, grid_out):
        try:
            # Read the binary stream into bytes
            data_bytes = grid_out.read()
            
            # Convert bytes into a string buffer
            data_io = io.BytesIO(data_bytes)
            
            # Read CSV into DataFrame
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            print(f"Error while converting grid_out to DataFrame: {e}")
            return None
        
    def check_and_convert_datatypes(self, df, datatype_dict):
        try:
            for column, expected_type in datatype_dict.items():
                if column not in df.columns:
                    continue

                # Track original empty/missing values
                original_empty_mask = df[column].isna() | (df[column].astype(str).str.strip() == '')

                if not self.is_same_datatype(df[column], expected_type):
                    df[column] = self.convert_to_datatype(df[column], expected_type)

                    # Drop only rows where conversion failed AND original wasn't empty
                    conversion_failed_mask = df[column].isna() & ~original_empty_mask
                    df = df[~conversion_failed_mask].copy()  # .copy() to avoid SettingWithCopyWarning

            return df
        except Exception as e:
            print(f"Error in checking and converting datatypes: {e}")
            return None

    def is_same_datatype(self, series, expected_type):
        non_null_series = series[series.notnull()]
        if expected_type == "int" and pd.api.types.is_integer_dtype(non_null_series):
            return True
        if expected_type == "float" and pd.api.types.is_float_dtype(non_null_series):
            return True
        if expected_type == "date_time" and pd.api.types.is_datetime64_any_dtype(non_null_series):
            return True
        if expected_type == "str" and pd.api.types.is_string_dtype(non_null_series):
            return True
        if expected_type == "bool" and pd.api.types.is_bool_dtype(non_null_series):
            return True
        if expected_type == "category" and pd.api.types.is_categorical_dtype(non_null_series):
            return True
        if expected_type == "time" and pd.api.types.is_datetime64_any_dtype(non_null_series):
            return True
        if expected_type == "timedelta" and pd.api.types.is_timedelta64_dtype(non_null_series):
            return True
        if expected_type == "complex" and pd.api.types.is_complex_dtype(non_null_series):
            return True
        if expected_type == "uint" and pd.api.types.is_unsigned_integer_dtype(non_null_series):
            return True
        if expected_type == "object":
            return True
        return False


    def convert_to_datatype(self, series, expected_type):
        try:
            if expected_type == "int":
                converted = pd.to_numeric(series, errors='coerce')
                return converted.astype('Int64')
            if expected_type == "float":
                return pd.to_numeric(series, errors='coerce')
            if expected_type == "date_time":
                # Convert to datetime without deprecated parameter
                dt_series = pd.to_datetime(series, errors='coerce')
                
                # Format to standardized string format after conversion
                return dt_series.dt.strftime("%Y-%m-%d") if not dt_series.empty else dt_series

            if expected_type == "time":
                # Convert to time objects, keep as datetime.time type
                return pd.to_datetime(series, errors='coerce').dt.time
            if expected_type == "str":
                return series.astype(str).replace({'nan': '', 'NaT': ''})
            if expected_type == "bool":
                series = series.astype(str).str.lower()
                try:
                    mapped_series = series.map({
                        'true': True, 'false': False,
                        '1': True, '0': False,
                        'yes': True, 'no': False
                    }).fillna(False)
                    return mapped_series
                except Exception:
                    unique_values = series.unique()
                    classified_dict = self.gemini_functions.get_gemini_classification_true_false(unique_values)
                    return series.map(classified_dict).fillna(False)

            if expected_type == "category":
                # Preprocess the series: convert to lowercase, strip spaces, handle NaNs
                processed_series = series.astype(str).str.strip().str.lower().replace('nan', np.nan)
                unique_values = processed_series[processed_series.notnull()].unique().tolist()
                
                # Get category mappings from Gemini
                category_mapping = self.gemini_functions.get_gemini_category_mapping(unique_values)
                
                # Map values to categories and convert to categorical type
                mapped_series = processed_series.map(category_mapping).astype("category")
                return mapped_series
            if expected_type == "timedelta":
                return pd.to_timedelta(series, errors='coerce')
            if expected_type == "complex":
                return series.apply(lambda x: complex(x) if isinstance(x, str) else x)
            if expected_type == "uint":
                return pd.to_numeric(series, errors='coerce').astype('UInt64')
            if expected_type == "object":
                return series.astype(object)
        except Exception as e:
            print(f"Error converting {series.name} to {expected_type}: {e}")
            return series

        
    def main(self):
        dataset_model = DatasetModel()
        dataset_details = dataset_model.get_dataset(self.dataset_id)
        if not dataset_details:
            print(f"Dataset with ID {self.dataset_id} not found.")
            return
        
        # Convert GridFS file to DataFrame
        grid_out = dataset_model.get_dataset_csv(self.dataset_id)
        df = self.gridout_to_dataframe(grid_out)
        expected_datatypes = dataset_details.get("datatype_of_each_column", {})

        # Check and convert datatypes
        df = self.check_and_convert_datatypes(df, expected_datatypes)
        if df is None:
            print("Error in converting dataset, returned empty df.")
            return
        

        

  