from flask import Flask
import os
from flask_cors import CORS
from app.cron.scheduler import init_scheduler
# from app.cron.model_training_scheduler import model_init_scheduler


def create_app():
    app = Flask(__name__)

    app.secret_key = os.getenv("SECRET_KEY", "your_default_secret_key")  # Ensure SECRET_KEY is set


    # Enable CORS
    CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

    # Register Blueprints
    from app.blueprints.users import users_bp
    from app.blueprints.datasets import datasets_bp
    from app.blueprints.model_building import model_building_bp
    from app.blueprints.gemini_apis import gemini_api_bp
    with app.app_context():
         print("Initializing scheduler...")
         init_scheduler(app)
         print("Scheduler initialized successfully!")

    app.register_blueprint(users_bp, url_prefix='/user')
    app.register_blueprint(datasets_bp, url_prefix='/dataset')
    app.register_blueprint(model_building_bp, url_prefix='/model')
    app.register_blueprint(gemini_api_bp, url_prefix='/gemini')

    with app.app_context():
        print("Initializing scheduler...")
        init_scheduler(app)
        print("Scheduler initialized successfully!")

    print("Registered URLs:")
    for rule in app.url_map.iter_rules():
        print(rule)
    return app
