#!/bin/bash

# Name of your image and container
IMAGE_NAME="thinkmap-backend"
CONTAINER_NAME="thinkmap-backend-container"

# Stop any existing container with the same name
echo "Stopping and removing existing container (if any)..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the container
echo "Starting Docker container: $CONTAINER_NAME..."
docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME

# Print container logs
echo "Tailing logs from $CONTAINER_NAME..."
docker logs -f $CONTAINER_NAME
