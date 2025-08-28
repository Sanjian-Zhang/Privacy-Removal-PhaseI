#!/usr/bin/env python3
"""
Convenient script for processing custom images
Supports automatic text detection or using provided coordinate files
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path

sys.path.append('./CODE')

def setup_directories(input_dir, output_dir, box_dir=None):
    """Create necessary directory structure"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if box_dir:
        Path(box_dir).mkdir(parents=True, exist_ok=True)
    return True

def detect_text_with_paddleocr(image_path, output_txt_path):
    """Use PaddleOCR to detect text regions and save coordinates"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        result = ocr.ocr(image_path, cls=True)
        
        with open(output_txt_path, 'w') as f:
            if result[0] is not None:
                for line in result[0]:
                    box = line[0]
                    coords = []
                    for point in box:
                        coords.extend([int(point[0]), int(point[1])])
                    f.write(','.join(map(str, coords)) + '\\n')
        
        return True
    except ImportError:
        print("❌ PaddleOCR not installed. Please run: pip install paddleocr")
        return False
    except Exception as e:
        print(f"❌ PaddleOCR detection failed: {e}")
        return False

def detect_text_with_opencv(image_path, output_txt_path):
    """Use OpenCV simple text detection (backup solution)"""
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(output_txt_path, 'w') as f:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:
                    coords = [x, y, x+w, y, x+w, y+h, x, y+h]
                    f.write(','.join(map(str, coords)) + '\\n')
        
        return True
    except Exception as e:
        print(f"❌ OpenCV detection failed: {e}")
        return False

def validate_data(image_dir, box_dir):
    """Validate correspondence between images and coordinate files"""
    image_files = set(Path(f).stem for f in glob.glob(os.path.join(image_dir, "*.jpg")))
    box_files = set(Path(f).stem for f in glob.glob(os.path.join(box_dir, "*.txt")))
    
    missing_boxes = image_files - box_files
    missing_images = box_files - image_files
    
    if missing_boxes:
        print(f"⚠️  Missing coordinate files: {missing_boxes}")
    if missing_images:
        print(f"⚠️  Missing image files: {missing_images}")
    
    valid_pairs = image_files & box_files
    print(f"✓ Found {len(valid_pairs)} valid image-coordinate file pairs")
    
    return len(valid_pairs) > 0

def run_garnet_inference(image_dir, box_dir, output_dir, gpu=True):
    """Run GaRNet inference"""
    cmd = f"""
    python CODE/inference.py \\
        --image_path {image_dir} \\
        --box_path {box_dir} \\
        --result_path {output_dir} \\
        --input_size 512 \\
        --model_path ./WEIGHTS/GaRNet/saved_model.pth
    """
    
    if gpu:
        cmd += " --gpu"
    
    print(f"🚀 Running GaRNet inference...")
    print(f"Command: {cmd}")
    
    return os.system(cmd) == 0

def main():
    parser = argparse.ArgumentParser(description="Process custom images - GaRNet text removal")
    parser.add_argument("--input_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output result directory")
    parser.add_argument("--box_dir", help="Coordinate file directory (will auto-detect if not provided)")
    parser.add_argument("--auto_detect", choices=["paddleocr", "opencv"], 
                       default="paddleocr", help="Automatic text detection method")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--validate_only", action="store_true", help="Only validate data format")
    
    args = parser.parse_args()
    
    print("🎯 GaRNet Image Processing Tool")
    print("=" * 50)
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    image_files = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_files:
        print(f"❌ No .jpg files found in {args.input_dir}")
        return
    
    print(f"✓ Found {len(image_files)} image files")
    
    if not args.box_dir:
        args.box_dir = os.path.join(os.path.dirname(args.input_dir), "auto_detected_boxes")
        print(f"📁 Will auto-detect text and save to: {args.box_dir}")
    
    setup_directories(args.input_dir, args.output_dir, args.box_dir)
    
    if not args.box_dir or not os.path.exists(args.box_dir) or len(glob.glob(os.path.join(args.box_dir, "*.txt"))) == 0:
        print(f"🔍 Auto-detecting text regions using {args.auto_detect}...")
        
        for image_file in image_files:
            image_name = Path(image_file).stem
            txt_file = os.path.join(args.box_dir, f"{image_name}.txt")
            
            print(f"   Processing: {image_name}")
            
            if args.auto_detect == "paddleocr":
                success = detect_text_with_paddleocr(image_file, txt_file)
            else:
                success = detect_text_with_opencv(image_file, txt_file)
            
            if not success:
                with open(txt_file, 'w') as f:
                    pass
                print(f"   ⚠️  Created empty coordinate file for {image_name}")
    
    print("\\n🔍 Validating data format...")
    if not validate_data(args.input_dir, args.box_dir):
        print("❌ Data validation failed")
        return
    
    if args.validate_only:
        print("✓ Data validation completed")
        return
    
    model_path = "./WEIGHTS/GaRNet/saved_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        print("Please run 'python download_model.py' to download the model first")
        return
    
    print("\\n🚀 Starting GaRNet text removal...")
    success = run_garnet_inference(args.input_dir, args.box_dir, args.output_dir, args.gpu)
    
    if success:
        print(f"\\n✅ Processing completed! Results saved at: {args.output_dir}")
        result_files = glob.glob(os.path.join(args.output_dir, "*.png"))
        print(f"📊 Generated {len(result_files)} result files")
    else:
        print("\\n❌ Processing failed")

if __name__ == "__main__":
    main()
