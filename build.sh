#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting FastAPI server with Uvicorn..."
uvicorn server:app --host 0.0.0.0 --port 8000
