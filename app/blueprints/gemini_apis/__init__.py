# Email feature component initialization file
# Organizes Gmail-related routes and functionality as a Flask Blueprint

from flask import Blueprint

# Create Blueprint instance for email-related routes
# - Name: 'emails' (used for URL routing)
# - Package: Current Python package (__name__)
gemini_api_bp = Blueprint('gemini_apis', __name__)

# Import views after blueprint creation to:
# 1. Avoid circular imports
# 2. Ensure blueprint object is available for route decorators
# 3. Maintain proper Flask application structure
from . import views  # Contains route definitions using gmail_emails_bp