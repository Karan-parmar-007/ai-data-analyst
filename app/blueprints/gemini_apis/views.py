from flask import request, jsonify
from app.blueprints.gemini_apis import gemini_api_bp
from app.models.datasets import DatasetModel
import google.generativeai as genai
import json
import pandas as pd
import io
import re

dataset_model = DatasetModel()

def gridout_to_dataframe(grid_out):
    try:
        data_bytes = grid_out.read()
        data_io = io.BytesIO(data_bytes)
        df = pd.read_csv(data_io)
        return df
    except Exception as e:
        print(f"Error converting grid_out to DataFrame: {e}")
        return None

@gemini_api_bp.route('/recommend_preprocessing', methods=['POST'])
def recommend_preprocessing():
    """
    API endpoint to fetch preprocessing recommendations from Gemini for all options.
    Expects JSON payload with dataset_id, user_goal, and target_column.
    Returns a JSON object with recommendations for all preprocessing steps.
    """
    # Get request data
    data = request.json
    dataset_id = data.get('dataset_id')
    target_column = data.get('target_column')

    # Validate input
    if not all([dataset_id, target_column]):
        return jsonify({"error": "Missing required parameters: dataset_id, user_goal, target_column"}), 400

    try:
        # Fetch dataset metadata
        dataset_info = dataset_model.get_dataset(dataset_id)
        if not dataset_info:
            return jsonify({"error": "Dataset not found"}), 404

        # Extract relevant fields
        column_datatypes = dataset_info.get('datatype_of_each_column', {})
        usage_of_each_column = dataset_info.get('usage_of_each_column', {})
        dataset_description = dataset_info.get('dataset_description', '')
        user_goal = dataset_info.get('what_user_wants_to_do', '')

        dataset_csv = dataset_model.get_dataset_csv(dataset_id)
        df = gridout_to_dataframe(dataset_csv)
        if df is None or df.empty:
            return jsonify({"error": "Failed to convert dataset to DataFrame"}), 500

        first_ten_rows = df.head(10)  
        shape = df.shape

        # Format columns for the prompt
        columns_str = ', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])

        # Construct prompt for Gemini
        prompt = f"""
            Dataset description: {dataset_description}
            Columns: {columns_str}
            User's goal: {user_goal}
            Target column: {target_column}
            description of each column: {usage_of_each_column}
            first ten rows: {first_ten_rows.to_string()}
            shape of the dataset: {shape}

            Please recommend the following preprocessing steps in JSON format:
            {{
            "fill_empty_rows_using": "mean" | "median" | "mode" | "none",
            "remove_duplicate": true | false,
            "standardization_necessary": true | false,
            "normalization_necessary": true | false,
            "test_dataset_percentage": integer (0-100),
            "increase_the_size_of_dataset": true | false,
            "fill_string_type_columns": true | false,
            "dimensionality_reduction": true | false,
            "remove_highly_correlated_columns": true | false
            }}
            """
        print(prompt)  # For debugging
        # Send request to Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')  # Adjust model name as needed
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        print(response_text)  # For debugging

        # Attempt to extract JSON object
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                recommendations = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Failed to parse extracted JSON: {json_str}")
                return jsonify({"error": "Invalid JSON response from Gemini"}), 500
        else:
            print(f"No JSON object found in response: {response_text}")
            return jsonify({"error": "No JSON object found in response"}), 500

        # Validate keys
        expected_keys = [
            "fill_empty_rows_using", "remove_duplicate", "standardization_necessary",
            "normalization_necessary", "test_dataset_percentage", "increase_the_size_of_dataset",
            "fill_string_type_columns", "dimensionality_reduction", "remove_highly_correlated_columns"
        ]
        if not all(key in recommendations for key in expected_keys):
            return jsonify({"error": "Incomplete recommendations from Gemini"}), 500

        return jsonify(recommendations)

    except Exception as e:
        print(f"Error in recommend_preprocessing: {e}")
        return jsonify({"error": "Internal server error"}), 500