from app import create_app
from dotenv import load_dotenv
import os
from flask import Flask


# Load environment variables from .env file
load_dotenv()

# Access environment variables (optional, to confirm they are loaded)
app = Flask(__name__)


app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')


print("Starting application...")


# Create the Flask app
app = create_app()


if __name__ == '__main__':
    # In production, you might want to set debug=False
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
