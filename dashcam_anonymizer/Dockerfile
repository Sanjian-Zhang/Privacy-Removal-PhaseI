FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory and download model
RUN mkdir -p model && \
    pip install gdown && \
    gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O model/best.pt

# Set environment variables
ENV PYTHONPATH=/app

# Create output directory
RUN mkdir -p /output

# Default command
CMD ["python", "blur_images.py", "--config", "configs/img_blur.yaml"]
