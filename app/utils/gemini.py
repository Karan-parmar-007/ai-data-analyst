from dotenv import load_dotenv
import os
from google import genai  # Updated import
from typing import List, Dict
import json
import logging
from app.models.datasets import DatasetModel
from app.models.classification_model import ClassificationModel
from app.models.model_hyperparameters import ModelHyperparameters
import pandas as pd
import io
import re
from flask import jsonify


dataset_model = DatasetModel()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiFunctions:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.dataset_model = DatasetModel()
        self.classification_model = ClassificationModel()
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        try:
            self.client = genai.Client(api_key=self.gemini_api_key)  # Create client instance
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            raise

    def get_gemini_classification_true_false(self, unique_values: List[str]) -> Dict[str, bool]:
        prompt = f"""
        Classify the following values as true or false.
        Respond only with JSON object where keys are the original values (lowercased) and values are true or false.
        Example:
        {{"yes": true, "no": false}}
        
        Values: {', '.join(unique_values)}
        """
        try:
            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
            response_text = response.text.strip()
            logger.info(f"Gemini Response: {response_text}")
            classified_dict = json.loads(response_text)
            if not isinstance(classified_dict, dict):
                raise ValueError("Gemini response is not a valid JSON object")
            return {k.lower(): bool(v) for k, v in classified_dict.items()}
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return {}

    def get_gemini_category_mapping(self, unique_values: List[str]) -> Dict[str, str]:
        prompt = f"""
        Map these values to standardized categories, correcting typos and unnecessary spaces.
        Respond only with JSON object where keys are the original values and values are standardized categories.
        Example:
        {{"fruits": "Fruit", "vegetbles": "Vegetable"}}
        
        Values: {', '.join(unique_values)}
        """
        try:
            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
            response_text = response.text.strip()
            logger.info(f"Gemini Response: {response_text}")
            
            # Remove markdown wrappers
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```
                response_text = response_text.strip()
            
            # Parse the cleaned JSON string
            category_dict = json.loads(response_text)
            if not isinstance(category_dict, dict):
                raise ValueError("Gemini response is not a valid JSON object")
            return category_dict
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return {}

    def gridout_to_dataframe(self, grid_out):
        try:
            data_bytes = grid_out.read()
            data_io = io.BytesIO(data_bytes)
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            print(f"Error converting grid_out to DataFrame: {e}")
            return None

    def get_type_of_models_to_built(self, dataset_id):
        if not dataset_id:
            return None
        try:
            # Fetch dataset metadata
            dataset_info = self.dataset_model.get_dataset(dataset_id)
            if not dataset_info:
                return None

            # Extract relevant fields
            column_datatypes = dataset_info.get('datatype_of_each_column', {})
            usage_of_each_column = dataset_info.get('usage_of_each_column', {})
            dataset_description = dataset_info.get('dataset_description', '')
            user_goal = dataset_info.get('what_user_wants_to_do', '')
            target_column = dataset_info.get('target_column', '')
            dataset_csv = self.dataset_model.get_dataset_csv(dataset_id)

            df = self.gridout_to_dataframe(dataset_csv)
            if df is None or df.empty:
                return None

            first_ten_rows = df.head(10)  
            shape = df.shape

            # Construct prompt for Gemini
            prompt = f'''
            You have been provided with the following dataset details:

            - **Dataset Description:** {dataset_description}
            - **Columns & Data Types:** {', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])}
            - **Column Usage Descriptions:** {usage_of_each_column}
            - **User's Goal:** {user_goal}
            - **Target Column:** {target_column}
            - **First 10 Rows of the Dataset:**  
            {first_ten_rows.to_string()}
            - **Dataset Shape (Rows, Columns):** {shape}

            Based on the above information, recommend the most suitable high-level model categories that should be used to achieve the given goal. 

            **Return the response as a JSON object in the following format:**
            
            {{
                "models": [
                    "classification_models",
                    "clustering"
                ]
            }}
            
            **DO NOT** include any specific models inside these categories.

            The available categories are:
            - "classification_models"
            - "clustering"
            - "naive_bayes"
            - "regression_models"
            '''

            print(prompt)  # Debugging

            # Send request to Gemini
            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
            response_text = response.text.strip()
            print(response_text)  # Debugging

            # Attempt to extract JSON object
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    recommendations = json.loads(json_str)
                    if isinstance(recommendations, dict) and "models" in recommendations:
                        return recommendations["models"]
                    else:
                        print(f"Unexpected JSON format: {json_str}")
                        return None
                except json.JSONDecodeError:
                    print(f"Failed to parse extracted JSON: {json_str}")
                    return None
            else:
                print(f"No JSON object found in response: {response_text}")
                return None
        
        except Exception as e:
            print(f"Error in get_type_of_models_to_built: {e}")
            return None  

    def get_type_of_models_for_one_type(self, dataset_id, models):
        if not dataset_id or not models or not isinstance(models, list) or len(models) == 0:
            return None
        if not all(isinstance(model, str) for model in models):
            return None
        
        try:
            # Fetch dataset metadata
            dataset_info = self.dataset_model.get_dataset(dataset_id)
            if not dataset_info:
                return None

            # Extract relevant fields
            column_datatypes = dataset_info.get('datatype_of_each_column', {})
            usage_of_each_column = dataset_info.get('usage_of_each_column', {})
            dataset_description = dataset_info.get('dataset_description', '')
            user_goal = dataset_info.get('what_user_wants_to_do', '')
            target_column = dataset_info.get('target_column', '')
            dataset_csv = self.dataset_model.get_dataset_csv(dataset_id)

            df = self.gridout_to_dataframe(dataset_csv)
            if df is None or df.empty:
                return None

            first_ten_rows = df.head(10)  
            shape = df.shape

            # Construct prompt for Gemini
            prompt = f'''
            You have been provided with the following dataset details:

            - **Dataset Description:** {dataset_description}
            - **Columns & Data Types:** {', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])}
            - **Column Usage Descriptions:** {usage_of_each_column}
            - **User's Goal:** {user_goal}
            - **Target Column:** {target_column}
            - **First 10 Rows of the Dataset:**  
            {first_ten_rows.to_string()}
            - **Dataset Shape (Rows, Columns):** {shape}

            Based on the above information, recommend the most suitable models from the given list that should be used to achieve the given goal. 

            **Only select models from the provided list and ensure they have at least a 60% accuracy expectation:**
            
            **Provided Model List:** {models}
            
            **Return the response as a JSON object in the following format:**
            
            {{
                "models": ["Linear Regression", "AdaBoost Classifier"]
            }}

            **Do not include any explanation or extra text. Only return the JSON object.**
            '''

            print(prompt)  # Debugging

            # Send request to Gemini
            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
            response_text = response.text.strip()
            print(response_text)  # Debugging

            # Attempt to extract JSON object
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    recommendations = json.loads(json_str)
                    if isinstance(recommendations, dict) and "models" in recommendations:
                        return recommendations["models"]
                    else:
                        print(f"Unexpected JSON format: {json_str}")
                        return None
                except json.JSONDecodeError:
                    print(f"Failed to parse extracted JSON: {json_str}")
                    return None
            else:
                print(f"No JSON object found in response: {response_text}")
                return None
        
        except Exception as e:
            print(f"Error in get_type_of_models_for_one_type: {e}")
            return None
        

    def recommend_models_for_dataset(self, dataset_id):
        if not dataset_id:
            print("No dataset_id provided.")
            return None

        try:
            # Step 1: Fetch all classification names and associated models
            classifications_with_models = self.classification_model.get_all_classifications_with_models()
            print("Fetched classifications with models:", classifications_with_models)

            if not classifications_with_models:
                print("No classifications with models found.")
                return None

            # Step 2: Fetch dataset metadata
            dataset_info = self.dataset_model.get_dataset(dataset_id)
            if not dataset_info:
                print("No dataset metadata found.")
                return None

            column_datatypes = dataset_info.get('datatype_of_each_column', {})
            usage_of_each_column = dataset_info.get('usage_of_each_column', {})
            dataset_description = dataset_info.get('dataset_description', '')
            user_goal = dataset_info.get('what_user_wants_to_do', '')
            target_column = dataset_info.get('target_column', '')
            dataset_csv = self.dataset_model.get_dataset_csv(dataset_id)
            df = self.gridout_to_dataframe(dataset_csv)

            if df is None or df.empty:
                print("DataFrame is empty or None.")
                return None

            first_ten_rows = df.head(10)
            shape = df.shape

            # Step 3: Ask Gemini for which classification types are suitable
            classification_prompt = f'''
            You have been provided with the following dataset details:

            - **Dataset Description:** {dataset_description}
            - **Columns & Data Types:** {', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])}
            - **Column Usage Descriptions:** {usage_of_each_column}
            - **User's Goal:** {user_goal}
            - **Target Column:** {target_column}
            - **First 10 Rows of the Dataset:**
            {first_ten_rows.to_string()}
            - **Dataset Shape (Rows, Columns):** {shape}

            Based on the above information, recommend the most suitable high-level model categories.

            Return a JSON object:
            {{ "models": ["Regression", "Classification"] }}

            Do not include any specific models. Only use these exact category names:
            - "Regression"
            - "Classification"
            - "Clustering"
            - "NaiveBayes"
            '''

            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=classification_prompt)
            print("Gemini classification response:", response)
            classification_response = response.text.strip()
            match = re.search(r'\{.*\}', classification_response, re.DOTALL)

            if not match:
                print("No JSON found in classification response")
                return None

            classification_json = json.loads(match.group(0))
            selected_classifications = classification_json.get("models", [])
            print("Selected classifications from Gemini:", selected_classifications)

            final_recommendations = {}

            # Step 4: For each selected classification, ask Gemini to pick relevant models
            for classification in selected_classifications:
                # Get the exact classification name from our database
                db_classification = next((k for k in classifications_with_models.keys() if k.lower() == classification.lower()), None)
                if not db_classification:
                    print(f"Classification {classification} not found in database")
                    continue

                model_list = list(classifications_with_models[db_classification].keys())
                print(f"Models available under classification '{db_classification}':", model_list)

                if not model_list:
                    continue

                model_prompt = f'''
                You have been provided with the following dataset details:

                - **Dataset Description:** {dataset_description}
                - **Columns & Data Types:** {', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])}
                - **Column Usage Descriptions:** {usage_of_each_column}
                - **User's Goal:** {user_goal}
                - **Target Column:** {target_column}
                - **First 10 Rows of the Dataset:**
                {first_ten_rows.to_string()}
                - **Dataset Shape (Rows, Columns):** {shape}

                From the given models list, select those expected to give more than 60% accuracy for achieving the user's goal.

                Provided Model List: {model_list}

                Return a JSON object:
                {{ "models": ["LinearRegression", "AdaBoostRegressor"] }}

                Do not include explanation. Only return models from the provided list.
                '''

                model_response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=model_prompt)
                model_response_text = model_response.text.strip()
                match = re.search(r'\{.*\}', model_response_text, re.DOTALL)

                if match:
                    try:
                        models_json = json.loads(match.group(0))
                        recommended_models = models_json.get("models", [])
                        # Filter to only include models that exist in our database
                        valid_models = [model for model in recommended_models if model in model_list]
                        if valid_models:
                            final_recommendations[db_classification] = valid_models
                            print(f"Recommended models for '{db_classification}':", valid_models)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON for classification {db_classification}")

            print("Final recommendations:", final_recommendations)

            if not final_recommendations:
                print("No preferred models were found after filtering.")
                return None

            return final_recommendations

        except Exception as e:
            print(f"Error in recommend_models_for_dataset: {e}")
            return None


    def get_hyperparameters(self, dataset_id, model_id):
        if not dataset_id or not model_id:
            return None

        try:
            # Fetch dataset metadata
            dataset_info = self.dataset_model.get_dataset(dataset_id)
            if not dataset_info:
                return None

            # Extract relevant fields
            column_datatypes = dataset_info.get('datatype_of_each_column', {})
            usage_of_each_column = dataset_info.get('usage_of_each_column', {})
            dataset_description = dataset_info.get('dataset_description', '')
            user_goal = dataset_info.get('what_user_wants_to_do', '')
            target_column = dataset_info.get('target_column', '')
            dataset_csv = self.dataset_model.get_dataset_csv(dataset_id)

            df = self.gridout_to_dataframe(dataset_csv)
            if df is None or df.empty:
                return None

            first_ten_rows = df.head(10)  
            shape = df.shape

            model_hyperparameters_model = ModelHyperparameters()
            model_hyperparameters = model_hyperparameters_model.get_hyperparameters_for_model(model_id)
            if not model_hyperparameters:
                print("No hyperparameters found for the model.")
                return None
            model_name = model_hyperparameters.get("model_name", "")
            all_hyperparameters = model_hyperparameters.get("hyperparameters", {})
            # Construct prompt for Gemini
            prompt = f'''
            You have been provided with the following dataset details:

            - **Dataset Description:** {dataset_description}
            - **Columns & Data Types:** {', '.join([f"{col}: {dtype}" for col, dtype in column_datatypes.items()])}
            - **Column Usage Descriptions:** {usage_of_each_column}
            - **User's Goal:** {user_goal}
            - **Target Column:** {target_column}
            - **First 10 Rows of the Dataset:**
            {first_ten_rows.to_string()}
            - **Dataset Shape (Rows, Columns):** {shape}

            Based on the above information, recommend the most all hyperparameters for the model '{model_name}'.

            Return a JSON object:
            
            {{
                "hyperparameters": {{
                    "param1": value1,
                    "param2": value2,
                    ...
                }}
            }}

            Do not include any explanation or extra text. Only return the JSON object.
            
            The available hyperparameters are:
            
                '{all_hyperparameters}'
                
                
                Please provide a detailed list of hyperparameters and their expected values.
                
                Example:
                
                {{
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "
                    "learning_rate": 0.01,
                    "alpha": 0.1
                }}
            '''
            print(prompt)  # Debugging
            # Send request to Gemini
            response = self.client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
            response_text = response.text.strip()
            print(response_text)  # Debugging
            # Attempt to extract JSON object
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    hyperparameters = json.loads(json_str)
                    if isinstance(hyperparameters, dict) and "hyperparameters" in hyperparameters:
                        return hyperparameters["hyperparameters"]
                    else:
                        print(f"Unexpected JSON format: {json_str}")
                        return None
                except json.JSONDecodeError:
                    print(f"Failed to parse extracted JSON: {json_str}")
                    return None
            else:
                print(f"No JSON object found in response: {response_text}")
                return None
        except Exception as e:
            print(f"Error in get_hyperparameters: {e}")
            return None