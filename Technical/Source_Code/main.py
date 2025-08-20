"""
Main application entry point for the Intelligence Platform.
"""

import os
import logging
from flask import Flask, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')


@app.route('/')
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Intelligence Platform API',
        'status': 'running',
        'version': '1.0.0'
    })


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'intelligence-platform'
    })


@app.route('/api/v1/status')
def api_status():
    """API status endpoint."""
    return jsonify({
        'api_version': 'v1',
        'status': 'operational',
        'features': [
            'data_processing',
            'analytics',
            'reporting',
            'machine_learning'
        ]
    })


if __name__ == '__main__':
    host = os.getenv('APP_HOST', '0.0.0.0')
    port = int(os.getenv('APP_PORT', 8000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Intelligence Platform on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
