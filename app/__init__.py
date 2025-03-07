from flask import Flask
import os
from flask_cors import CORS
from app.cron.scheduler import init_scheduler


def create_app():
    app = Flask(__name__)

    app.secret_key = os.getenv("SECRET_KEY", "your_default_secret_key")  # Ensure SECRET_KEY is set


    # Enable CORS
    CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

    # Register Blueprints
    from app.blueprints.users import users_bp
    from app.blueprints.datasets import datasets_bp

    app.register_blueprint(users_bp, url_prefix='/user')
    app.register_blueprint(datasets_bp, url_prefix='/dataset')

    with app.app_context():
        print("Initializing scheduler...")
        init_scheduler(app)
        print("Scheduler initialized successfully!")

    print("Registered URLs:")
    for rule in app.url_map.iter_rules():
        print(rule)
    return app
