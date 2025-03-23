from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets import DatasetModel
from app.utils.column_conversion import ColumnConversion
from app.utils.handle_null_values import HandleNullValues
from app.utils.remove_duplicates import RemoveDuplicates
from datetime import datetime
import pandas as pd
import numpy as np

_scheduler = None 

def preprocess_dataset():
    """Job to process only unpreprocessed datasets with standardization."""
    try:
        dataset_model = DatasetModel()
        all_datasets = list(dataset_model.datasets_collection.find())
        
        if not all_datasets:
            print("No datasets found in the database.")
            return

        # Check if all datasets are preprocessed
        all_preprocessed = all(dataset.get("is_preprocessing_done", False) for dataset in all_datasets)
        if all_preprocessed:
            print("All datasets are already preprocessed. Skipping preprocessing job.")
            return

        # Process only unpreprocessed datasets
        unpreprocessed_datasets = [d for d in all_datasets if not d.get("is_preprocessing_done", False)]
        for dataset in unpreprocessed_datasets:
            dataset_id = str(dataset["_id"])
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

                # Update dataset with standardized data and set status
                dataset_model.update_dataset_file(dataset_id, df, is_preprocessing_done=True)
                print(f"Dataset {dataset_id} preprocessed successfully.")
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")
    except Exception as e:
        print(f"Error in preprocess_dataset: {str(e)}")

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