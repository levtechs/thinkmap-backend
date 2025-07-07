#!/bin/bash

# Optional: echo startup info
echo "Starting FastAPI server..."
echo "Running on http://0.0.0.0:8000"

# Start FastAPI using uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
