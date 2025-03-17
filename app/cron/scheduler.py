from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.models.datasets import DatasetModel
from app.utils.column_conversion import ColumnConversion
from app.utils.handle_null_values import HandleNullValues
from app.utils.remove_duplicates import RemoveDuplicates
from datetime import datetime
import pandas as pd
import numpy as np

_scheduler = None 

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame columns to have mean=0 and std=1."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()
    stds = stds.replace(0, 1)
    df[numeric_cols] = (df[numeric_cols] - means) / stds
    return df

def preprocess_dataset():
    """Job to process new unprocessed datasets with standardization."""
    try:
        datasets_to_be_preprocessed_model = DatasetsToBePreprocessedModel()
        dataset_model = DatasetModel()
        unprocessed_datasets = datasets_to_be_preprocessed_model.get_unprocessed_datasets()
        
        for dataset_id in unprocessed_datasets:
            try:
                # Load initial dataset
                grid_out = dataset_model.get_dataset_csv(dataset_id)
                handle_null = HandleNullValues(dataset_id)
                df = handle_null.gridout_to_dataframe(grid_out)
                if df is None or df.empty:
                    continue

                # Step 1: Column Conversion
                process_column = ColumnConversion(dataset_id=dataset_id)
                if not process_column.main():
                    continue
                grid_out = dataset_model.get_dataset_csv(dataset_id)
                df = handle_null.gridout_to_dataframe(grid_out)

                # Step 2: Handle Null Values
                handle_null_values = HandleNullValues(dataset_id=dataset_id)
                if not handle_null_values.main():
                    continue
                grid_out = dataset_model.get_dataset_csv(dataset_id)
                df = handle_null.gridout_to_dataframe(grid_out)

                # Step 3: Remove Duplicates
                remove_duplicates = RemoveDuplicates(dataset_id=dataset_id)
                if not remove_duplicates.main():
                    continue
                grid_out = dataset_model.get_dataset_csv(dataset_id)
                df = handle_null.gridout_to_dataframe(grid_out)

                # Step 4: Standardize the DataFrame
                df = standardize_dataframe(df)

                # Update dataset with standardized data and set status
                dataset_model.update_dataset_file(dataset_id, df, is_preprocessing_done=True)
                datasets_to_be_preprocessed_model.delete_dataset_to_be_preprocessed(dataset_id)
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")  # Temporary debug
    except Exception as e:
        print(f"Error in preprocess_dataset: {str(e)}")  # Temporary debug

def init_scheduler(app):
    """Initialize the scheduler with the Flask app context."""
    global _scheduler
    
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return
        
    if _scheduler is not None:
        return
        
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        func=preprocess_dataset,
        trigger='interval',
        minutes=5,
        id='preprocess_dataset_job',
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now()
    )
    _scheduler.start()
    
    import atexit
    atexit.register(lambda: _scheduler.shutdown(wait=False))