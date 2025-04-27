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
from app.models.classification_model import ClassificationModel
from app.models.trained_model import TrainedModel


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
    classification_model = ClassificationModel()
    trained_model = TrainedModel()
    unbuilt_models = model_to_be_built.get_all_models()
    
    # Define model name mappings for each category
    clustering_model_mapping = {
        "agglomerative_clustering": "AgglomerativeClustering",
        "kmeans": "KMeans",
        "dbscan": "DBSCAN",
        "birch_clustering": "Birch",
        "k_means_clustering": "KMeans",
        "mean_shift": "MeanShift",
        "mini_batch_kmeans": "MiniBatchKMeans",
        "spectral_clustering": "SpectralClustering"
    }
    
    regression_model_mapping = {
        "linear_regression": "LinearRegression",
        "lasso": "Lasso",
        "ridge": "Ridge",
        "elastic_net": "ElasticNet",
        "adaboost_regressor": "AdaBoostRegressor",
        "support_vector_regression": "SVR"
    }
    
    naive_bayes_model_mapping = {
        "gaussian_nb": "GaussianNB",
        "bernoulli_nb": "BernoulliNB",
        "categorical_nb": "CategoricalNB",
        "complement_nb": "ComplementNB",
        "multinomial_nb": "MultinomialNB"
    }
    
    classification_model_mapping = {
        "knn_classifier": "KNeighborsClassifier",
        "logistic_regression": "LogisticRegression",
        "decision_tree_classifier": "DecisionTreeClassifier",
        "random_forest_classifier": "RandomForestClassifier",
        "gradient_boosting_classifier": "GradientBoostingClassifier",
        "adaboost_classifier": "AdaBoostClassifier",
        "svc": "SVC"
    }
    
    for model in unbuilt_models:
        try:
            if not isinstance(model, dict) or "dataset_id" not in model or "classification_id" not in model:
                continue
                
            dataset_id = model["dataset_id"]
            classification_id = model["classification_id"]
            
            classification = classification_model.get_all_classifications_with_models()
            classification_name = None
            for cls in classification["classifications"]:
                if cls["_id"] == classification_id:
                    classification_name = cls["classification_name"]
                    break
            
            if not classification_name:
                continue
                
            data_transformation = DataTransformation(dataset_id)
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation()
            
            if all(val is not None for val in (X_train, X_test, y_train, y_test)):
                # Get data transformation file ID
                dataset = dataset_model.get_dataset(dataset_id)
                data_transformation_pkl_id = dataset.get("data_transformation_file_and_transformaed_dataset", {}).get("config_file_id")
                
                if not data_transformation_pkl_id:
                    continue
                
                if classification_name.lower() == "classification":
                    from app.utils.train_and_store_model.classidication_models_training import ClassificationModeTraining
                    trainer = ClassificationModeTraining(dataset_id, X_train, X_test, y_train, y_test)
                    for cls in classification["classifications"]:
                        if cls["_id"] == classification_id:
                            for model_data in cls["models"]:
                                # Map model name if needed
                                model_name = model_data["model_name"]
                                if model_name.lower() in classification_model_mapping:
                                    model_name = classification_model_mapping[model_name.lower()]
                                
                                try:
                                    result = trainer.train_model(model_name, model_data["hyperparameters"])
                                    if result and isinstance(result, dict) and "artifact_store_response" in result:
                                        # Store all model information in trained_models collection
                                        trained_model.create_model(
                                            dataset_id=dataset_id,
                                            model_name=model_data["model_name"],
                                            classification_name=classification_name,
                                            hyperparameters=model_data["hyperparameters"],
                                            data_transformation_pkl_id=str(data_transformation_pkl_id),
                                            trained_model_pkl_id=str(result["artifact_store_response"]["model_file_id"]),
                                            train_time=result.get("train_time", 0),
                                            metrics=result.get("metrics", {}),
                                            progress=result.get("progress", [])
                                        )
                                        # Delete from model_building collection
                                        model_to_be_built.delete_model(model["_id"])
                                except ValueError as e:
                                    print(f"Error training {model_name} for dataset {dataset_id}: {str(e)}")
                

                elif classification_name.lower() == "clustering":
                    from app.utils.train_and_store_model.clustering_models_training import ClusteringModelTraining
                    trainer = ClusteringModelTraining(dataset_id, X_train, X_test, y_train, y_test)
                    for cls in classification["classifications"]:
                        if cls["_id"] == classification_id:
                            for model_data in cls["models"]:
                                # Map model name if needed
                                model_name = model_data["model_name"]
                                if model_name.lower() in clustering_model_mapping:
                                    model_name = clustering_model_mapping[model_name.lower()]
                                
                                try:
                                    result = trainer.train_model(model_name, model_data["hyperparameters"])
                                    if result and isinstance(result, dict) and "artifact_store_response" in result:
                                        # Handle the case where artifact_id might be an ObjectId
                                        artifact_id = result["artifact_store_response"]
                                        if hasattr(artifact_id, '__str__'):
                                            artifact_id = str(artifact_id)
                                        elif isinstance(artifact_id, dict) and "model_file_id" in artifact_id:
                                            artifact_id = str(artifact_id["model_file_id"])
                                        
                                        # Store all model information in trained_models collection
                                        trained_model.create_model(
                                            dataset_id=dataset_id,
                                            model_name=model_data["model_name"],
                                            classification_name=classification_name,
                                            hyperparameters=model_data["hyperparameters"],
                                            data_transformation_pkl_id=str(data_transformation_pkl_id),
                                            trained_model_pkl_id=artifact_id,
                                            train_time=result.get("train_time", 0),
                                            metrics=result.get("metrics", {}),
                                            progress=result.get("progress", [])
                                        )
                                        # Delete from model_building collection
                                        model_to_be_built.delete_model(model["_id"])
                                except ValueError as e:
                                    print(f"Error training {model_name} for dataset {dataset_id}: {str(e)}")

                
                elif classification_name.lower() == "naivebayes" or classification_name.lower() == "naive bayes":
                    from app.utils.train_and_store_model.naive_bayes_model_tranining import NaiveBayesModelTraining
                    trainer = NaiveBayesModelTraining(dataset_id, X_train, X_test, y_train, y_test)
                    for cls in classification["classifications"]:
                        if cls["_id"] == classification_id:
                            for model_data in cls["models"]:
                                # Map model name if needed
                                model_name = model_data["model_name"]
                                if model_name.lower() in naive_bayes_model_mapping:
                                    model_name = naive_bayes_model_mapping[model_name.lower()]
                                
                                try:
                                    result = trainer.train_model(model_name, model_data["hyperparameters"])
                                    if result and isinstance(result, dict) and "artifact_store_response" in result:
                                        # Store all model information in trained_models collection
                                        trained_model.create_model(
                                            dataset_id=dataset_id,
                                            model_name=model_data["model_name"],
                                            classification_name=classification_name,
                                            hyperparameters=model_data["hyperparameters"],
                                            data_transformation_pkl_id=str(data_transformation_pkl_id),
                                            trained_model_pkl_id=str(result["artifact_store_response"]["model_file_id"]),
                                            train_time=result.get("train_time", 0),
                                            metrics=result.get("metrics", {}),
                                            progress=result.get("progress", [])
                                        )
                                        # Delete from model_building collection
                                        model_to_be_built.delete_model(model["_id"])
                                except ValueError as e:
                                    print(f"Error training {model_name} for dataset {dataset_id}: {str(e)}")
                
                elif classification_name.lower() == "regression":
                    from app.utils.train_and_store_model.regression_model_training import RegressionModelTraining
                    trainer = RegressionModelTraining(dataset_id, X_train, X_test, y_train, y_test)
                    # Use the model name directly from the unbuilt_models collection
                    model_name = model.get("model_name")
                    hyperparameters = model.get("hyperparameters", {})
                    
                    # Train the model with the exactly specified model name
                    try:
                        result = trainer.train_model(model_name, hyperparameters)
                        if result and isinstance(result, dict) and "artifact_store_response" in result:
                            # Store all model information in trained_models collection
                            trained_model.create_model(
                                dataset_id=dataset_id,
                                model_name=model_name, 
                                classification_name=classification_name,
                                hyperparameters=hyperparameters,
                                data_transformation_pkl_id=str(data_transformation_pkl_id),
                                trained_model_pkl_id=str(result["artifact_store_response"]["model_file_id"]),
                                train_time=result.get("train_time", 0),
                                metrics=result.get("metrics", {}),
                                progress=result.get("progress", [])
                            )
                            # Delete from model_building collection - make sure we're using a string ID
                            model_id = model.get("_id")
                            if model_id:
                                model_to_be_built.delete_model(model_id)
                    except Exception as e:
                        print(f"Error training model {model_name} for dataset {dataset_id}: {str(e)}")
        except Exception as e:
            print(f"Error building model for dataset {dataset_id if dataset_id else 'unknown'}: {str(e)}")

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