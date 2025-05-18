#!/bin/bash

echo "==================================================="
echo " Object Detection Application - Setup and Run"
echo "==================================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.8 or newer."
    echo "On Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "On macOS: brew install python3"
    exit 1
fi

echo "Python is installed. Installing dependencies..."

# Create a virtual environment (optional)
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing required packages..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "==================================================="
echo " Setup complete! Starting the application..."
echo "==================================================="

# Run the Flask application
export FLASK_APP=app.py
export FLASK_DEBUG=1
python3 -m flask run

# Deactivate virtual environment when the app is closed
deactivate 