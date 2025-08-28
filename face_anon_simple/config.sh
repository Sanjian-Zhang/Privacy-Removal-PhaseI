#!/bin/bash

# Face Anonymization Simple Configuration
# Edit these paths according to your environment

# ===========================================
# PATH CONFIGURATION
# ===========================================

# Project root directory (where the face_anon_simple folder is located)
export PROJECT_ROOT="/home/zhiqics/sanjian/testdocker/face_anon_simple"

# Input directory containing images to process
export INPUT_DIR="/home/zhiqics/sanjian/test_dataset/images/Train"

# Output directory for processed images
export OUTPUT_DIR="/home/zhiqics/sanjian/output/face_anon_results"

# ===========================================
# GPU CONFIGURATION
# ===========================================

# Batch size - adjust based on your GPU memory:
# 48GB GPU: batch_size=4 (default)
# 24GB GPU: batch_size=2
# 16GB GPU: batch_size=1
# 12GB GPU: batch_size=1
export BATCH_SIZE=4

# ===========================================
# DOCKER CONFIGURATION
# ===========================================

# Docker image name and tag
export IMAGE_NAME="face-anon-processor"
export IMAGE_TAG="v3"

# Container name
export CONTAINER_NAME="face-anon-processing"

# Docker runtime (nvidia for standard Docker, or use --gpus all)
export DOCKER_RUNTIME="--runtime=nvidia --gpus all"

# ===========================================
# PROCESSING PARAMETERS
# ===========================================

# Anonymization degree (0.0 = face swap, 1.25 = anonymization)
export ANONYMIZATION_DEGREE=1.25

# Guidance scale
export GUIDANCE_SCALE=4.0

# Number of inference steps
export NUM_INFERENCE_STEPS=25

# ===========================================
# SYSTEM CHECK FUNCTIONS
# ===========================================

check_paths() {
    echo "Checking configuration..."
    
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "ERROR: Project root does not exist: $PROJECT_ROOT"
        echo "Please edit PROJECT_ROOT in config.sh"
        return 1
    fi
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "ERROR: Input directory does not exist: $INPUT_DIR"
        echo "Please edit INPUT_DIR in config.sh"
        return 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    
    return 0
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
        return 1
    fi
    
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    return 0
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found. Please install Docker."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "ERROR: Docker daemon not running or no permission."
        echo "Try: sudo systemctl start docker"
        return 1
    fi
    
    echo "Docker is available"
    return 0
}

# Run all checks
run_system_check() {
    echo "=========================================="
    echo "Face Anonymization Simple - System Check"
    echo "=========================================="
    
    check_paths && check_gpu && check_docker
    
    if [ $? -eq 0 ]; then
        echo "✓ System check passed!"
        echo "You can now run: ./quick_start_face_anon.sh"
    else
        echo "✗ System check failed. Please fix the issues above."
    fi
}
