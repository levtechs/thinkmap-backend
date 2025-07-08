# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for scipy, numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY server.py .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variable for production
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 8080

# Use environment variable for port, fallback to 8080
CMD ["python", "server.py"]