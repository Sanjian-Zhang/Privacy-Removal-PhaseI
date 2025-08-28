#!/bin/bash

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <input_directory> <output_directory> [face|fullbody]"
    echo "Example: $0 /path/to/input/dir /path/to/output/dir face"
    echo "Default mode: face"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MODE="${3:-face}"

if [ "$MODE" != "face" ] && [ "$MODE" != "fullbody" ]; then
    echo "Error: Mode must be 'face' or 'fullbody'"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting batch anonymization..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Processing mode: $MODE"

if [ "$MODE" = "face" ]; then
    CONFIG="configs/anonymizers/face.py"
else
    CONFIG="configs/anonymizers/FB_cse.py"
fi

sudo docker run --rm \
  -v "$INPUT_DIR":/home/zhiqics/input \
  -v "$OUTPUT_DIR":/home/zhiqics/output \
  -v /home/zhiqics/sanjian/testdocker/deep_privacy2:/home/zhiqics/deep_privacy2 \
  -w /home/zhiqics/deep_privacy2 \
  deep_privacy2 \
  bash -c "
    echo 'Starting batch processing...'
    count=0
    total=\$(find ../input -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l)
    echo \"Found \$total images\"
    
    for img in ../input/*.{jpg,jpeg,png,bmp}; do
        if [ -f \"\$img\" ]; then
            filename=\$(basename \"\$img\")
            output_file=\"../output/anonymized_\$filename\"
            
            echo \"Processing: \$filename (\$((++count))/\$total)\"
            
            python3 anonymize.py $CONFIG \
                -i \"\$img\" \
                --output_path \"\$output_file\"
            
            if [ \$? -eq 0 ]; then
                echo \"✅ Completed: \$filename\"
            else
                echo \"❌ Failed: \$filename\"
            fi
        fi
    done
    echo 'Batch processing completed!'
  "

echo "✅ Batch anonymization processing completed!"
echo "Results saved in: $OUTPUT_DIR"
