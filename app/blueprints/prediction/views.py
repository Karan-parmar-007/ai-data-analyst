from flask import request, jsonify
from app.models.datasets import DatasetModel
from app.blueprints.prediction import prediction_bp
from app.models.trained_model import TrainedModel
import pandas as pd
import numpy as np
from app.utils.db import db
import gridfs
import pickle
from bson import ObjectId

@prediction_bp.route('/predict_new_value', methods=['POST'])
def predict_new_value():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")
        model_id = data.get("model_id")
        new_data = data.get("new_data")

        if not dataset_id or not model_id or not new_data:
            return jsonify({"error": "dataset_id, model_id, and new_data are required"}), 400

        trained_model = TrainedModel()
        model = trained_model.get_model_by_id(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        # Create GridFS instance for file retrieval
        fs = gridfs.GridFS(db)
        
        # Get the trained model file from GridFS
        trained_model_pkl_id = model.get("trained_model_pkl_id")
        if not trained_model_pkl_id:
            return jsonify({"error": "Model file ID not found"}), 404
        
        # Get the data transformation file from GridFS
        data_transformation_pkl_id = model.get("data_transformation_pkl_id")
        if not data_transformation_pkl_id:
            return jsonify({"error": "Data transformation file ID not found"}), 404
        
        # Load the trained model from GridFS
        try:
            model_file = fs.get(ObjectId(trained_model_pkl_id))
            trained_model_obj = pickle.loads(model_file.read())
        except Exception as e:
            return jsonify({"error": f"Failed to load trained model: {str(e)}"}), 500
        
        # Load the data transformation object from GridFS
        try:
            transform_file = fs.get(ObjectId(data_transformation_pkl_id))
            preprocessor = pickle.loads(transform_file.read())
        except Exception as e:
            return jsonify({"error": f"Failed to load data transformation: {str(e)}"}), 500
        
        # Convert new_data to a pandas DataFrame
        try:
            # Create a DataFrame with a single row from the input data dictionary
            input_df = pd.DataFrame([new_data])
            
            # Apply the preprocessor transformation to the input data
            transformed_input = preprocessor.transform(input_df)
            
            # Make prediction using the trained model
            prediction = trained_model_obj.predict(transformed_input)
            
            # Convert numpy array to Python list for JSON serialization
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            # Return the prediction result
            return jsonify({
                "prediction": prediction[0] if len(prediction) == 1 else prediction,
                "model_name": model.get("model_name", "Unknown model"),
                "classification_name": model.get("classification_name", "Unknown classification")
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Prediction failed: {str(e)}",
                "details": "Ensure that input data contains all required features and correct data types"
            }), 400
            
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
        