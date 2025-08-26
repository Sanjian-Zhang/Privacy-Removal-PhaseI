# Garnet Text Anonymizer Dockerfile - Train Dataset Version
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy garnet source code
COPY garnet_code/ /workspace/

# Install Python dependencies (CPU version for compatibility)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy opencv-python ptflops scikit-image pillow

# Download garnet model
RUN pip install huggingface-hub
RUN python download_model.py

# Create necessary directories
RUN mkdir -p /input /output /workspace/result

# Copy the train processing script
COPY process_train.py /workspace/

# Set the entry point
ENTRYPOINT ["python", "process_train.py"]
