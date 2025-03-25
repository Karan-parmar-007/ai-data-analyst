import os
import sys
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
from bson import ObjectId
import io
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from app.models.datasets import DatasetModel  # Import DatasetModel for artifact storage
from app.utils.exception import CustomException  # Ensure this exception prints the underlying error
from app.utils.logger import logging



def convert_datetime(x, col):
    """Top-level function for datetime conversion (must be outside the class)."""
    # Convert input to numpy array and flatten
    x_flat = x.to_numpy().ravel()
    sample = x_flat[0] if len(x_flat) > 0 else None

    try:
        if sample and isinstance(sample, str):
            if ':' in sample and '-' not in sample:
                # Handle time-only format
                dt_series = pd.to_datetime(x_flat, format='%H:%M:%S', errors='coerce')
            else:
                # Handle full datetime
                dt_series = pd.to_datetime(x_flat, errors='coerce')
        else:
            dt_series = pd.to_datetime(x_flat, errors='coerce')
    except Exception as e:
        logging.error(f"Error converting datetime column {col}: {e}")
        dt_series = pd.to_datetime(x_flat, errors='coerce')

    # Convert to Series to ensure .dt accessor works
    dt_series = pd.Series(dt_series, name=col)

    # Fill missing values with a default datetime
    dt_series = dt_series.fillna(pd.Timestamp("1900-01-01"))

    # Extract datetime components using .dt
    df_out = pd.DataFrame({
        f"{col}_day": dt_series.dt.day,
        f"{col}_month": dt_series.dt.month,
        f"{col}_year": dt_series.dt.year,
        f"{col}_hour": dt_series.dt.hour,
        f"{col}_minute": dt_series.dt.minute,
        f"{col}_second": dt_series.dt.second,
    })

    return df_out.values


class DataTransformation:
    def __init__(self, dataset_id: str):
        """
        Initializes the data transformation process by retrieving the dataset from the database.
        The dataset document (metadata) should include:
           - "datatype_of_each_column": dict with column names as keys and expected types (e.g., 'float', 'datetime', etc.)
           - "target_column": name of target column (if any)
           - flags such as "standardization_necessary", "normalization_necessary", "dimensionality_reduction", etc.
           - "test_dataset_percentage": percentage for test split (e.g., 30 for 30% test split)
           - "_id": the MongoDB _id for the dataset (stored as ObjectId)
        """
        self.ds_model = DatasetModel()
        dataset_data = self.ds_model.get_dataset(ObjectId(dataset_id))
        if not dataset_data:
            raise CustomException(f"No dataset found with ID: {dataset_id}", sys)
        
        grid_out = self.ds_model.get_dataset_csv(dataset_id)
        data_bytes = grid_out.read()
        data_io = io.BytesIO(data_bytes)
        self.df = pd.read_csv(data_io)
        # Store the entire dataset document as metadata.
        self.metadata = dataset_data

    def _build_datetime_pipeline(self, col: str):
        return Pipeline(steps=[
            ('datetime_extractor', FunctionTransformer(
                convert_datetime,  # Use the top-level function
                validate=False,
                kw_args={"col": col}  # Pass column name here
            ))
        ])



    def get_preprocessing_pipeline(self):
        try:
            logging.info("Building preprocessing pipeline based on dataset metadata.")
            # Use the full document from metadata.
            datatype_dict = self.metadata.get("datatype_of_each_column", {})
            target_col = self.metadata.get("target_column", "")

            num_cols = []
            cat_cols = []
            datetime_cols = []
            
            # Separate columns based on expected data types.
            for col, dtype in datatype_dict.items():
                if col == target_col:
                    continue  # Skip target column.
                dtype = dtype.lower()
                if dtype in ["float", "double", "int", "uint", "integer"]:
                    num_cols.append(col)
                elif dtype in ["category", "object", "str"]:
                    cat_cols.append(col)
                elif dtype in ["datetime", "date", "time"]:
                    datetime_cols.append(col)

            # Build numerical pipeline.
            num_pipeline_steps = [
                ('imputer', SimpleImputer(strategy='median'))
            ]
            if self.metadata.get("standardization_necessary", False):
                num_pipeline_steps.append(('scaler', StandardScaler()))
            elif self.metadata.get("normalization_necessary", False):
                num_pipeline_steps.append(('scaler', MinMaxScaler()))
            num_pipeline = Pipeline(steps=num_pipeline_steps)

            # Build categorical pipeline.
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])


            transformers = []
            if num_cols:
                transformers.append(('num_pipeline', num_pipeline, num_cols))
            if cat_cols:
                transformers.append(('cat_pipeline', cat_pipeline, cat_cols))
            if datetime_cols:
                for col in datetime_cols:
                    transformers.append((f"dt_pipeline_{col}", self._build_datetime_pipeline(col), [col]))

            preprocessor = ColumnTransformer(transformers=transformers)

            if self.metadata.get("dimensionality_reduction", False):
                preprocessor = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('pca', PCA(n_components=0.95))
                ])

            logging.info("Preprocessing pipeline built successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error while building the preprocessing pipeline: %s", e)
            raise CustomException(e, sys)

    def transform_dataset(self, df: pd.DataFrame):
        """
        Applies the preprocessing pipeline to the provided DataFrame.
        Returns the fitted preprocessor and the transformed data.
        """
        try:
            preprocessor = self.get_preprocessing_pipeline()
            logging.info("Transforming dataset using the preprocessor pipeline.")
            transformed_data = preprocessor.fit_transform(df)
            return preprocessor, transformed_data
        except Exception as e:
            logging.error("Error during dataset transformation: %s", e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Reads the dataset from GridFS (already loaded in self.df), applies the preprocessing pipeline,
        performs train-test split based on metadata, and stores the preprocessor pipeline (as a pickle file)
        in GridFS by updating the dataset document field "data_transformation_file_and_transformaed_dataset".
        
        Returns X_train, X_test, y_train, y_test.
        """
        try:
            # Use the DataFrame already loaded in __init__.
            df = self.df.copy()

            # Extract target column if provided.
            target_col = self.metadata.get("target_column", "")
            if target_col and target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
            else:
                X = df
                y = None

            # Apply transformation on features.
            preprocessor, transformed_data = self.transform_dataset(X)
            transformed_df = pd.DataFrame(transformed_data)
            logging.info("Dataset transformation complete. Sample output: %s", str(transformed_df.head()))

            # Perform train-test split based on metadata percentage.
            test_pct = self.metadata.get("test_dataset_percentage", 30) / 100.0
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(transformed_df, y, test_size=test_pct, random_state=42)
            else:
                X_train, X_test = train_test_split(transformed_df, test_size=test_pct, random_state=42)
                y_train = y_test = None

            # Retrieve the dataset ID (using _id field from metadata).
            dataset_id = str(self.metadata.get("_id"))
            if not dataset_id:
                raise CustomException("Dataset ID not found in metadata.", sys)

            # Store only the preprocessor pipeline artifact.
            # Note: We pass an empty DataFrame to indicate that we're not storing the transformed dataset.
            artifacts = self.ds_model.store_artifacts(dataset_id, preprocessor)
            logging.info("Artifacts stored in DB: %s", artifacts)
            print(X_train, X_test, y_train, y_test)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Exception occurred during data transformation: %s", e)
            raise CustomException(e, sys)
