from apscheduler.schedulers.background import BackgroundScheduler
import os
from app.models.datasets_to_be_preprocessed import DatasetsToBePreprocessedModel
from app.utils.dataset_preprocessing import DatasetPreprocessing
from datetime import datetime

_scheduler = None 


# def preprocess_dataset():
#     """Job to process new unprocessed emails."""
#     print(f"[{datetime.now()}] Starting process_new_emails job...")
#     try:
#         # Initialize the DatasetPreprocessing class
#         dataset_preprocessing = DatasetPreprocessing()
#         datasets_to_be_preprocessed_model = DatasetsToBePreprocessedModel()

#         # Get all unprocessed datasets
#         unprocessed_datasets = datasets_to_be_preprocessed_model.get_unprocessed_datasets()
#         print(f"[{datetime.now()}] Found {len(unprocessed_datasets)} unprocessed datasets")
#         for dataset in unprocessed_datasets:
#             # Preprocess the dataset
#             dataset_preprocessing.main(dataset)
#             # Delete the dataset from the unprocessed list
#             datasets_to_be_preprocessed_model.delete_dataset_to_be_preprocessed(dataset)
#         print(f"[{datetime.now()}] process_new_emails job completed successfully")
#     except Exception as e:
#         print(f"[{datetime.now()}] Error in process_new_emails job: {str(e)}")

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
                dataset_preprocessing = DatasetPreprocessing(dataset_id=dataset_id)
                # Preprocess the dataset
                dataset_preprocessing.main()
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