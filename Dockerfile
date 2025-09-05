# ===== Lean Dockerfile for production =====
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install system dependencies, Python packages, then remove build tools in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libgomp1 \
        gfortran \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential cmake gfortran \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy app code (excluding raw data via .dockerignore)
COPY . .

# Expose port only if needed for testing
EXPOSE 7860

# Start the app
CMD ["python", "app.py"]
