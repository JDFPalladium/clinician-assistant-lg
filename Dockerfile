# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Create and set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with retries
RUN echo "Installing Python dependencies..." && \
    pip install --no-cache-dir --verbose -r requirements.txt


# Copy application files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]