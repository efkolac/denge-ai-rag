# Use the official PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_DIR=/model \
    HF_HOME=/hf_cache \
    TORCH_HOME=/torch_cache

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /app /context_files ${MODEL_DIR} ${HF_HOME} ${TORCH_HOME}

# Set working directory (must come before COPY commands)
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY context_files ./context_files/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify files are in place
RUN ls -la /app

# Run the application
CMD ["python", "-u", "app.py"]