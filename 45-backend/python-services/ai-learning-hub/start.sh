#!/bin/bash

echo "🤖 Starting AI Learning Hub Service..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Start the service
echo "🚀 Starting service on port 5020..."
python app.py
