from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.models.datasets import DatasetModel
from app.utils.column_conversion import ColumnConversion
from app.utils.handle_null_values import HandleNullValues
from app.utils.remove_duplicates import RemoveDuplicates
from datetime import datetime

_scheduler = None 

def preprocess_dataset():
    """Job to process new unprocessed datasets."""
    try:
        datasets_to_be_preprocessed_model = DatasetsToBePreprocessedModel()
        dataset_model = DatasetModel()
        unprocessed_datasets = datasets_to_be_preprocessed_model.get_unprocessed_datasets()
        
        for dataset_id in unprocessed_datasets:
            try:
                process_column = ColumnConversion(dataset_id=dataset_id)
                if not process_column.main():
                    continue
                
                handle_null_values = HandleNullValues(dataset_id=dataset_id)
                if not handle_null_values.main():
                    continue

                remove_duplicates = RemoveDuplicates(dataset_id=dataset_id)
                if not remove_duplicates.main():
                    continue    

                # Update preprocessing status
                dataset_model.update_preprocessing_status(dataset_id, is_preprocessing_done=True)
                datasets_to_be_preprocessed_model.delete_dataset_to_be_preprocessed(dataset_id)
            except Exception:
                pass  # Silent error handling
    except Exception:
        pass  # Silent error handling

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