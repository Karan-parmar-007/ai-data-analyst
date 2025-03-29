from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets import DatasetModel
from app.utils.preprocesssing_part_one.column_conversion import ColumnConversion
from app.utils.preprocesssing_part_one.handle_null_values import HandleNullValues
from app.utils.preprocesssing_part_one.remove_duplicates import RemoveDuplicates
from datetime import datetime
import pandas as pd
import numpy as np
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.models.model_building import ModelToBeBuilt
from app.utils.preprocessing_part_two.dataset_model_preprocessing import DataTransformation


_scheduler = None 

def preprocess_dataset():
    """Job to process only unpreprocessed datasets with standardization."""
    try:
        datasets_to_be_preprocessed_model = DatasetsToBePreprocessedModel()
        dataset_model = DatasetModel()
        unprocessed_datasets = datasets_to_be_preprocessed_model.get_unprocessed_datasets()

        for dataset_id in unprocessed_datasets:
            print(f"Processing dataset with ID: {dataset_id}")  # ✅ Now dataset_id is correct

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

                dataset_model.stop_preprocessing(dataset_id)
                dataset_model.update_dataset_file(dataset_id, df, is_preprocessing_done=True)
                datasets_to_be_preprocessed_model.delete_dataset_to_be_preprocessed(dataset_id)
                print(f"Dataset {dataset_id} preprocessed successfully.")
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")
    except Exception as e:
        print(f"Error in preprocess_dataset: {str(e)}")

def start_model_building():
    model_to_be_built = ModelToBeBuilt()
    dataset_model = DatasetModel()
    unbuilt_datasets = model_to_be_built.get_unbuilt_model()
    for dataset_id in unbuilt_datasets:
        try:
            data_transformation = DataTransformation(dataset_id)
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation()
            if all(val is not None for val in (X_train, X_test, y_train, y_test)):
                model_to_be_built.delete_model(dataset_id)
                print(f"Dataset {dataset_id} model built successfully.")
        except Exception as e:
            print(f"Error building model for dataset {dataset_id}: {str(e)}")




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

    _scheduler.add_job(
        func=start_model_building,
        trigger='interval',
        minutes=5,
        id='start_model_building_job',
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now()
    )

    _scheduler.start()
    
    import atexit
    atexit.register(lambda: _scheduler.shutdown(wait=False))