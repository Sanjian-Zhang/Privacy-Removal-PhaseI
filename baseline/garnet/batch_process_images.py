#!/usr/bin/env python3
"""
Batch image processing script for GaRNet text removal
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import torch
import subprocess

# 添加CODE目录到路径
garnet_path = "/home/zhiqics/sanjian/baseline/garnet"
sys.path.append(os.path.join(garnet_path, "CODE"))

def setup_environment():
    """Setup environment and activate virtual environment"""
    print("Setting up GaRNet environment...")
    
    # Check if we're in the garnet directory
    if not os.path.exists(os.path.join(garnet_path, "WEIGHTS/GaRNet/saved_model.pth")):
        print(f"Error: GaRNet model not found at {garnet_path}")
        print("Please make sure GaRNet is properly set up.")
        return False
    
    return True

def detect_text_regions(image_path):
    """
    Simple text region detection
    Uses OpenCV EAST text detector or simple edge detection
    In real applications, you may need to use more advanced text detection models
    """
    print(f"Detecting text regions in {image_path}...")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # Simple example: create some dummy text boxes
    # In real applications, you need to use actual text detection algorithms
    text_boxes = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out too small regions (may not be text)
        if w > 30 and h > 10 and w < width * 0.8 and h < height * 0.3:
            # Convert rectangle to 4-point format (x1,y1,x2,y2,x3,y3,x4,y4)
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            
            text_boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    
    # If no text detected, create a demo text box for demonstration
    if not text_boxes:
        # Create a demo text box at image center
        center_x, center_y = width // 2, height // 2
        box_width, box_height = min(200, width // 3), min(50, height // 6)
        
        x1 = center_x - box_width // 2
        y1 = center_y - box_height // 2
        x2 = center_x + box_width // 2
        y2 = center_y - box_height // 2
        x3 = center_x + box_width // 2
        y3 = center_y + box_height // 2
        x4 = center_x - box_width // 2
        y4 = center_y + box_height // 2
        
        text_boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        print(f"No text detected, created demo box at center")
    
    print(f"Detected {len(text_boxes)} text regions")
    return text_boxes

def create_text_file(image_path, text_boxes, output_dir):
    """Create corresponding text box coordinate file"""
    image_name = Path(image_path).stem
    txt_file = os.path.join(output_dir, f"{image_name}.txt")
    
    with open(txt_file, 'w') as f:
        for box in text_boxes:
            # Format: x1,y1,x2,y2,x3,y3,x4,y4
            line = ','.join(map(str, box))
            f.write(line + '\n')
    
    print(f"Created text file: {txt_file}")
    return txt_file

def process_images_with_garnet(input_dir, output_dir, temp_img_dir, temp_txt_dir):
    """Process images using GaRNet"""
    print("Processing images with GaRNet...")
    
    # Switch to garnet directory
    original_cwd = os.getcwd()
    os.chdir(garnet_path)
    
    try:
        # Activate virtual environment and run inference
        cmd = [
            "bash", "-c",
            f"source garnet_env/bin/activate && "
            f"cd CODE && "
            f"python inference.py --gpu "
            f"--image_path {temp_img_dir} "
            f"--box_path {temp_txt_dir} "
            f"--result_path {output_dir}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ GaRNet processing completed successfully")
            print("STDOUT:", result.stdout)
        else:
            print("✗ GaRNet processing failed")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        print(f"Error running GaRNet: {e}")
        return False
    finally:
        os.chdir(original_cwd)
    
    return True

def main():
    """Main function"""
    input_dir = "/home/zhiqics/sanjian/dataset/test_images/images"
    output_dir = "/home/zhiqics/sanjian/dataset/test_images/anon/garnet"
    
    print("="*60)
    print("GaRNet Batch Text Removal Processing")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check environment
    if not setup_environment():
        return False
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directories for processed images and text files
    temp_base = "/tmp/garnet_processing"
    temp_img_dir = os.path.join(temp_base, "images")
    temp_txt_dir = os.path.join(temp_base, "txt")
    
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_txt_dir, exist_ok=True)
    
    try:
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print("No image files found in input directory")
            return False
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_file}...")
            
            input_image_path = os.path.join(input_dir, image_file)
            temp_image_path = os.path.join(temp_img_dir, image_file)
            
            # Copy image to temporary directory
            shutil.copy2(input_image_path, temp_image_path)
            
            # Detect text regions
            text_boxes = detect_text_regions(input_image_path)
            
            # Create text coordinate file
            create_text_file(temp_image_path, text_boxes, temp_txt_dir)
        
        # Use GaRNet for batch processing
        if process_images_with_garnet(input_dir, output_dir, temp_img_dir, temp_txt_dir):
            print(f"\n✓ All images processed successfully!")
            print(f"✓ Results saved at: {output_dir}")
            
            # List processed files
            if os.path.exists(output_dir):
                processed_files = os.listdir(output_dir)
                print(f"✓ Generated {len(processed_files)} processed files:")
                for file in processed_files:
                    print(f"  - {file}")
        else:
            print("✗ Processing failed")
            return False
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_base):
            shutil.rmtree(temp_base)
            print("Cleaned up temporary files")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Batch processing completed successfully!")
    else:
        print("\n❌ Batch processing failed")
        sys.exit(1)
