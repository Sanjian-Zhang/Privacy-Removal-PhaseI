#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_image_path> <output_image_path>"
    echo "Example: $0 /path/to/input.jpg /path/to/output.jpg"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="$2"

INPUT_DIR=$(dirname "$INPUT_PATH")
INPUT_FILE=$(basename "$INPUT_PATH")
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
OUTPUT_FILE=$(basename "$OUTPUT_PATH")

mkdir -p "$OUTPUT_DIR"

echo "Starting full-body anonymization..."
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"

sudo docker run --rm \
  -v "$INPUT_DIR":/home/zhiqics/input \
  -v "$OUTPUT_DIR":/home/zhiqics/output \
  -v /home/zhiqics/sanjian/testdocker/deep_privacy2:/home/zhiqics/deep_privacy2 \
  -w /home/zhiqics/deep_privacy2 \
  deep_privacy2 \
  python3 anonymize.py configs/anonymizers/FB_cse.py \
  -i "../input/$INPUT_FILE" \
  --output_path "../output/$OUTPUT_FILE" \
  --visualize

if [ $? -eq 0 ]; then
    echo "✅ Full-body anonymization completed: $OUTPUT_PATH"
else
    echo "❌ Full-body anonymization failed"
    exit 1
fi
