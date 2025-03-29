from dotenv import load_dotenv
import os
import google.generativeai as genai  # Corrected import
from typing import List, Dict
import json
import logging
from app.models.datasets import DatasetModel
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
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        try:
            genai.configure(api_key=self.gemini_api_key)
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
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
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
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
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
            model = genai.GenerativeModel('gemini-1.5-flash')  # Adjust model name as needed
            response = model.generate_content(prompt)
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



    def gridout_to_dataframe(self, grid_out):
        try:
            data_bytes = grid_out.read()
            data_io = io.BytesIO(data_bytes)
            df = pd.read_csv(data_io)
            return df
        except Exception as e:
            print(f"Error converting grid_out to DataFrame: {e}")
            return None
        

    