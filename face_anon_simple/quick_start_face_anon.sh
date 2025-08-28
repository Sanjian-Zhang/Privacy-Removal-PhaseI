#!/bin/bash

# Face Anonymization Simple Quick Start Script
# Usage: ./quick_start.sh [input_directory] [output_directory]

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    echo "Warning: config.sh not found, using default values"
    # Default paths (fallback)
    DEFAULT_INPUT="/home/zhiqics/sanjian/test_dataset/images/Train"
    DEFAULT_OUTPUT="/home/zhiqics/sanjian/output/face_anon_results"
fi

# Use parameters or configured values
INPUT_DIR=${1:-${INPUT_DIR:-$DEFAULT_INPUT}}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-$DEFAULT_OUTPUT}}

echo "=========================================="
echo "Face Anonymization Simple Quick Start"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check GPU status
echo "Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Stop and remove old containers if they exist
echo "Cleaning up old containers..."
sudo docker stop $CONTAINER_NAME 2>/dev/null || true
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

# Check if image exists
IMAGE_FULL_NAME="${IMAGE_NAME:-face-anon-processor}:${IMAGE_TAG:-v3}"
if [ -z "$(sudo docker images -q $IMAGE_FULL_NAME)" ]; then
    echo "Image does not exist, building..."
    cd "$SCRIPT_DIR"
    sudo docker build -t $IMAGE_FULL_NAME .
fi

# Run container
echo "Starting face anonymization processing..."
CONTAINER_NAME="${CONTAINER_NAME:-face-anon-processing}"
sudo docker run ${DOCKER_RUNTIME:---gpus all} \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    --name $CONTAINER_NAME \
    $IMAGE_FULL_NAME

# Display results
echo "=========================================="
echo "Processing completed!"
echo "Check output directory: $OUTPUT_DIR"
echo "View logs: sudo docker logs $CONTAINER_NAME"
echo "=========================================="

# Display output directory contents
echo "Output files:"
ls -la "$OUTPUT_DIR" | head -10
