"""
VIT Media Vault — AI Microservice
===================================
Entry point for the Flask API server.

This server loads the CLIP AI model at startup, then listens for
image processing requests from the React frontend.

Usage:
  cd ai-service
  .\\venv\\Scripts\\activate      (Windows)
  pip install -r requirements.txt
  python app.py

The server runs on http://localhost:5000
"""
from flask import Flask
from flask_cors import CORS

from config import FLASK_PORT
from models.clip_model import load_model
from routes.process import process_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Allow the React frontend (localhost:5173) to call this API
    CORS(app, origins=[r"http://localhost:\d+", r"http://127\.0\.0\.1:\d+"])
    
    # Register API routes
    app.register_blueprint(process_bp)
    
    return app


# Create the Flask app globally for WSGI servers like Gunicorn
app = create_app()

# In production, Gunicorn will skip the __main__ block
# We no longer load the model in the global scope to save boot RAM
# load_model()

if __name__ == "__main__":
    # Local runs load eagerly so the first upload isn't stuck loading the model mid-request
    load_model()

    print(f"\n{'='*60}")
    print(f"  VIT Media Vault — AI Engine")
    print(f"  Running on http://localhost:{FLASK_PORT}")
    print(f"  Features: Auto-Tagging | Deduplication | Semantic Search")
    print(f"{'='*60}\n")
    
    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=False  # Debug=False because model loading is heavy
    )
