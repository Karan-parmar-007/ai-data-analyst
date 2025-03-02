from app.blueprints.gemini_apis import gemini_api_bp
from flask import Flask, request, jsonify
from app.models.users import UserModel
from app.models.datasets import DatasetModel
from bson.errors import InvalidId

user_model = UserModel()
dataset_model = DatasetModel()


