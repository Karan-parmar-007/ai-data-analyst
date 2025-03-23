from app.models.datasets import DatasetModel
import pandas as pd
import io
import numpy as np
from pandas.api.types import is_integer_dtype, is_float_dtype, is_datetime64_any_dtype, is_string_dtype, is_bool_dtype, is_categorical_dtype, is_timedelta64_dtype, is_complex_dtype, is_unsigned_integer_dtype
from app.utils.gemini import GeminiFunctions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColumnConversion:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.gemini_functions = GeminiFunctions()
        # Dictionary mapping expected types to conversion functions
        self.conversion_functions = {
            "int": self.convert_to_int,
            "float": self.convert_to_float,
            "date_time": self.convert_to_datetime,
            "str": self.convert_to_str,
            "bool": self.convert_to_bool,
            "category": self.convert_to_category,
            "time": self.convert_to_time,
            "timedelta": self.convert_to_timedelta,
            "complex": self.convert_to_complex,
            "uint": self.convert_to_uint,
            "object": self.convert_to_object,
        }

    def gridout_to_dataframe(self, grid_out):
        try:
            data_bytes = grid_out.read()
            data_io = io.BytesIO(data_bytes)
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            logger.error(f"Error converting grid_out to DataFrame: {e}")
            return None

    def check_and_convert_datatypes(self, df, datatype_dict):
        try:
            for column, expected_type in datatype_dict.items():
                if column not in df.columns:
                    continue
                if not self.is_same_datatype(df[column], expected_type):
                    conversion_function = self.conversion_functions.get(expected_type)
                    if conversion_function:
                        original_na_count = df[column].isna().sum()
                        df[column] = conversion_function(df[column])
                        new_na_count = df[column].isna().sum()
                        if new_na_count > original_na_count:
                            logger.warning(f"Conversion to {expected_type} introduced {new_na_count - original_na_count} new NaN values in column {column}")
                    else:
                        logger.warning(f"No conversion function for {expected_type} in column {column}")
            return df
        except Exception as e:
            logger.error(f"Error in checking and converting datatypes: {e}")
            return None

    def is_same_datatype(self, series, expected_type):
        non_null_series = series[series.notnull()]
        if expected_type == "int" and is_integer_dtype(non_null_series):
            return True
        if expected_type == "float" and is_float_dtype(non_null_series):
            return True
        if expected_type == "date_time" and is_datetime64_any_dtype(non_null_series):
            return True
        if expected_type == "str" and is_string_dtype(non_null_series):
            return True
        if expected_type == "bool" and is_bool_dtype(non_null_series):
            return True
        if expected_type == "category" and is_categorical_dtype(non_null_series):
            return True
        if expected_type == "time" and series.dtype == 'object' and non_null_series.apply(lambda x: isinstance(x, pd.Timestamp)).all():
            return True
        if expected_type == "timedelta" and is_timedelta64_dtype(non_null_series):
            return True
        if expected_type == "complex" and is_complex_dtype(non_null_series):
            return True
        if expected_type == "uint" and is_unsigned_integer_dtype(non_null_series):
            return True
        if expected_type == "object":
            return True
        return False

    ### Conversion Functions ###
    def convert_to_int(self, series):
        """Convert series to nullable integer type, handling non-numeric values."""
        try:
            converted = pd.to_numeric(series, errors='coerce')
            return converted.astype('Int64')  # Nullable integer type to preserve NaNs
        except Exception as e:
            logger.error(f"Error converting {series.name} to int: {e}")
            return series

    def convert_to_float(self, series):
        """Convert series to float, preserving NaNs for non-numeric values."""
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            logger.error(f"Error converting {series.name} to float: {e}")
            return series

    def convert_to_datetime(self, series):
        """Convert series to datetime, formatting as string, handling invalid dates."""
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
            return dt_series.dt.strftime("%Y-%m-%d") if not dt_series.empty else dt_series
        except Exception as e:
            logger.error(f"Error converting {series.name} to datetime: {e}")
            return series

    def convert_to_str(self, series):
        """Convert series to string, replacing NaN/NaT with empty strings."""
        try:
            return series.astype(str).replace({'nan': '', 'NaT': ''})
        except Exception as e:
            logger.error(f"Error converting {series.name} to str: {e}")
            return series

    def convert_to_bool(self, series):
        """Convert series to boolean, using mapping and Gemini for complex cases."""
        try:
            if is_bool_dtype(series):
                return series
            series = series.astype(str).str.lower().str.strip()
            simple_mapping = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}
            if series.isin(simple_mapping.keys()).all():
                return series.map(simple_mapping).fillna(False)
            unique_values = [v for v in series.unique() if v and v not in simple_mapping]
            if unique_values:
                classified_dict = self.gemini_functions.get_gemini_classification_true_false(unique_values)
                full_mapping = {**simple_mapping, **classified_dict}
                return series.map(full_mapping).fillna(False)
            return series.map(simple_mapping).fillna(False)
        except Exception as e:
            logger.error(f"Error converting {series.name} to bool: {e}")
            return series

    def convert_to_category(self, series):
        """Convert series to category, using Gemini for mapping with fallback to original values."""
        try:
            if is_categorical_dtype(series):
                return series
            
            processed_series = series.astype(str).str.strip().str.lower().replace('nan', np.nan)
            unique_values = processed_series[processed_series.notnull()].unique().tolist()
            
            if not unique_values:
                return processed_series.astype("category")
            
            # Get Gemini mapping
            category_mapping = self.gemini_functions.get_gemini_category_mapping(unique_values)
            print(f"this is gemini response :{category_mapping}")
            # Ensure all unique values have a mapping, fallback to original value if not found
            for value in unique_values:
                if value not in category_mapping:
                    category_mapping[value] = value  # Retain original value if unmapped
            
            # Map values and convert to category
            categorized_series = processed_series.map(category_mapping).astype("category")
            
            return categorized_series
        
        except Exception as e:
            logger.error(f"Error converting {series.name} to category: {e}")
            return series

    def convert_to_time(self, series):
        """Convert series to time objects, extracting time from datetime."""
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
            return dt_series.dt.time
        except Exception as e:
            logger.error(f"Error converting {series.name} to time: {e}")
            return series

    def convert_to_timedelta(self, series):
        """Convert series to timedelta, handling invalid formats."""
        try:
            return pd.to_timedelta(series, errors='coerce')
        except Exception as e:
            logger.error(f"Error converting {series.name} to timedelta: {e}")
            return series

    def convert_to_complex(self, series):
        """Convert series to complex numbers, handling valid inputs only."""
        try:
            return series.apply(lambda x: complex(x) if isinstance(x, (str, int, float)) else x)
        except Exception as e:
            logger.error(f"Error converting {series.name} to complex: {e}")
            return series

    def convert_to_uint(self, series):
        """Convert series to unsigned integer, preserving NaNs."""
        try:
            converted = pd.to_numeric(series, errors='coerce')
            return converted.astype('UInt64')
        except Exception as e:
            logger.error(f"Error converting {series.name} to uint: {e}")
            return series

    def convert_to_object(self, series):
        """Return series as object type (default pandas type)."""
        return series

    def main(self):
        dataset_model = DatasetModel()
        dataset_details = dataset_model.get_dataset(self.dataset_id)
        if not dataset_details:
            logger.error(f"Dataset with ID {self.dataset_id} not found.")
            return
        
        grid_out = dataset_model.get_dataset_csv(self.dataset_id)
        df = self.gridout_to_dataframe(grid_out)
        if df is None or df.empty:
            logger.error(f"Failed to convert dataset {self.dataset_id} to DataFrame or DataFrame is empty.")
            return
        
        expected_datatypes = dataset_details.get("datatype_of_each_column", {})
        df = self.check_and_convert_datatypes(df, expected_datatypes)
        if df is None:
            logger.error(f"Error converting datatypes for dataset {self.dataset_id}.")
            return
        print(df)
        print("after column conversion")
        logger.info(f"Dataset {self.dataset_id} preprocessed successfully.")
        # Optionally save the preprocessed DataFrame back to the database here
        dataset_model.update_dataset_file(self.dataset_id, df, is_preprocessing_done=False)
        return True