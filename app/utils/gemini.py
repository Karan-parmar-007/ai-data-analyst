from dotenv import load_dotenv
import os
import google.generativeai as genai  # Corrected import
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GeminiFunctions:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
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