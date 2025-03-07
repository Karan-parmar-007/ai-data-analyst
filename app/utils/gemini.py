from dotenv import load_dotenv
import os
from google import genai
from typing import List, Dict
import json

load_dotenv()

class GeminiFunctions:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=self.gemini_api_key)

    def get_gemini_classification_true_false(self, unique_values: List[str]) -> Dict[str, bool]:
        prompt = f"""
        Classify the following values as true or false.
        Respond only with JSON object where keys are the original values (lowercased) and values are true or false.
        Example:
        {{"yes": true, "no": false}}
        
        Values: {', '.join(unique_values)}
        """
        try:
            model = genai.GenerativeModel("gemini-1.0-pro")
            response = model.generate_content(prompt)
            response_text = response.text
            print("Gemini Response:", response_text)
            classified_dict = json.loads(response_text)
            return classified_dict
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
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
            model = genai.GenerativeModel("gemini-1.0-pro")
            response = model.generate_content(prompt)
            response_text = response.text
            print("Gemini Response:", response_text)
            category_dict = json.loads(response_text)
            return category_dict
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return {}
