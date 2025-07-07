#!/bin/bash
# build.sh

# Optional: Debugging output
echo "Starting FastAPI server..."

# Start the server
exec uvicorn server:app --host 0.0.0.0 --port 8000
