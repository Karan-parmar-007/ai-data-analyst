from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.utils.column_conversion import ColumnConversion
from app.utils.handle_null_values import HandleNullValues
from app.utils.remove_duplicates import RemoveDuplicates
from datetime import datetime

_scheduler = None 

def preprocess_dataset():
    """Job to process new unprocessed datasets."""
    print(f"[{datetime.now()}] Starting process_new_emails job...")
    try:
        # Initialize models
        datasets_to_be_preprocessed_model = DatasetsToBePreprocessedModel()

        # Get all unprocessed datasets (assumed to be a list of dataset IDs)
        unprocessed_datasets = datasets_to_be_preprocessed_model.get_unprocessed_datasets()
        print(f"[{datetime.now()}] Found {len(unprocessed_datasets)} unprocessed datasets")
        
        for dataset_id in unprocessed_datasets:  # Directly use the ID string
            try:
                # Initialize DatasetPreprocessing with the dataset ID
                process_column = ColumnConversion(dataset_id=dataset_id)
                # Preprocess the dataset
                status_of_column_processing = process_column.main()

                if not status_of_column_processing:
                    print(f"Error processing dataset {dataset_id}")
                    continue
                
                handle_null_values = HandleNullValues(dataset_id=dataset_id)
                status_of_null_values_removed = handle_null_values.main()
                if not status_of_null_values_removed:
                    print(f"Error processing dataset {dataset_id}")
                    continue

                remove_duplicates = RemoveDuplicates(dataset_id=dataset_id)
                status_of_duplicates_removed = remove_duplicates.main()
                if not status_of_duplicates_removed:
                    print(f"Error processing dataset {dataset_id}")
                    continue    

                # Delete the dataset from the unprocessed list
                datasets_to_be_preprocessed_model.delete_dataset_to_be_preprocessed(dataset_id)
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")
        print(f"[{datetime.now()}] process_new_emails job completed successfully")
    except Exception as e:
        print(f"[{datetime.now()}] Error in process_new_emails job: {str(e)}")

def init_scheduler(app):
    """Initialize the scheduler with the Flask app context."""
    global _scheduler
    
    # Check if we're in the reloader process
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("Skipping scheduler initialization in reloader process")
        return
        
    if _scheduler is not None:
        print(f"[{datetime.now()}] Scheduler already initialized!")
        return
        
    print(f"[{datetime.now()}] Creating new scheduler instance...")
    _scheduler = BackgroundScheduler()

    _scheduler.add_job(
        func=preprocess_dataset,
        trigger='interval',
        minutes=5,
        id='process_new_emails_job',
        max_instances=1,
        coalesce=True,  # Combine multiple waiting runs into a single run
        next_run_time=datetime.now()  # Start the first run immediately
    )

    print(f"[{datetime.now()}] Starting scheduler...")
    _scheduler.start()
    print(f"[{datetime.now()}] Scheduler started successfully!")
    
    # Shut down scheduler when app terminates
    import atexit
    atexit.register(lambda: _scheduler.shutdown(wait=False))


