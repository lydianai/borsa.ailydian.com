"""
SIGNAL GENERATOR SERVICE
Real-time trading signals from 14 AI models
Port: 5004
White-hat compliant - transparent, fair signals
"""

import sys
import os

# Add the virtual environment's site-packages to the path
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'python3.14', 'site-packages')
sys.path.insert(0, venv_path)

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import requests
import time
import threading
from datetime import datetime
from consensus_engine import ConsensusEngine

# Prometheus client import with error handling
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    prometheus_available = True
except ImportError:
    print("Warning: Prometheus client not available")
    prometheus_available = False

# Force unbuffered output
sys.stdout = sys.stderr

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Service URLs
AI_SERVICE_URL = "http://localhost:5003"
DATA_SERVICE_URL = "http://localhost:3000"

# Consensus engine
consensus_engine = ConsensusEngine()

# Prometheus metrics (if available)
if prometheus_available:
    signal_requests_total = Counter('signal_requests_total', 'Total number of signal requests')
    signal_processing_time = Histogram('signal_processing_time', 'Time spent processing signals')
    active_signals = Gauge('active_signals', 'Number of active signals')

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        "service": "Signal Generator Service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

# Metrics endpoint (if Prometheus is available)
@app.route('/metrics')
def metrics():
    if prometheus_available:
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    else:
        return jsonify({"error": "Prometheus not available"}), 503

# Main application
if __name__ == '__main__':
    print("🚀 Signal Generator Service starting...")
    print(f"📡 Listening on port 5004")
    print(f"🤖 AI Service URL: {AI_SERVICE_URL}")
    print(f"📊 Data Service URL: {DATA_SERVICE_URL}")
    
    app.run(host='0.0.0.0', port=5004, debug=False)