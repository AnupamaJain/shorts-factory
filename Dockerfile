# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for MoviePy, FFmpeg, and OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the outputs and temp directories exist
RUN mkdir -p outputs temp assets/bgm assets/icons assets/broll inputs/channelvideo inputs/rag_data

# Set the environment variable for Python path
ENV PYTHONPATH=/app

# Command to keep the container running or run a specific script
# Users can override this to run agents
CMD ["python", "agents/graph.py", "Introduction to AI Agents"]
