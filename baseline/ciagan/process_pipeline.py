#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIAGAN one-click processing script - from original images to anonymized results
"""

import os
import sys
import argparse
import subprocess
import shutil

def process_images_pipeline(input_dir, output_dir):
    """
    Complete image processing pipeline
    """
    print("=== CIAGAN Image Processing Pipeline ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return False
    
    # Check image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"❌ No image files found in input directory")
        return False
    
    print(f"✅ Found {len(image_files)} images")
    
    # Create temporary directories
    temp_processed = os.path.join(output_dir, "temp_processed")
    final_output = os.path.join(output_dir, "anonymized")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    try:
        # Step 1: Preprocess images
        print("\n=== Step 1: Preprocessing images ===")
        cmd1 = [
            "python", "process_test_images.py",
            "--input", input_dir,
            "--output", temp_processed
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            print(f"❌ Preprocessing failed: {result1.stderr}")
            return False
        
        print("✅ Image preprocessing completed")
        
        # Step 2: Fix filename format
        print("\n=== Step 2: Fixing filename format ===")
        identity_dir = os.path.join(temp_processed, "clr", "identity_0")
        if os.path.exists(identity_dir):
            # Rename identity_0 to 0
            new_identity_dir = os.path.join(temp_processed, "clr", "0")
            os.rename(identity_dir, new_identity_dir)
            
            # Perform same operation on all subdirectories
            for subdir in ['lndm', 'msk', 'orig']:
                old_path = os.path.join(temp_processed, subdir, "identity_0")
                new_path = os.path.join(temp_processed, subdir, "0")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            
            # Rename files to 6-digit format
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                subdir_path = os.path.join(temp_processed, subdir, "0")
                if os.path.exists(subdir_path):
                    files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                    for i, filename in enumerate(files):
                        old_file = os.path.join(subdir_path, filename)
                        new_file = os.path.join(subdir_path, f"{i:06d}.jpg")
                        os.rename(old_file, new_file)
        
        print("✅ Filename format fixing completed")
        
        # Step 3: Run identity anonymization
        print("\n=== Step 3: Running identity anonymization ===")
        cmd2 = [
            "python", "source/test.py",
            "--data", temp_processed + "/",
            "--model", "pretrained_models/modelG",
            "--out", final_output + "/",
            "--ids", "1"
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"❌ Identity anonymization failed: {result2.stderr}")
            return False
        
        print("✅ Identity anonymization completed")
        
        # Clean up temporary files
        if os.path.exists(temp_processed):
            shutil.rmtree(temp_processed)
        
        # Check results
        result_files = [f for f in os.listdir(final_output) if f.endswith('.jpg')]
        print(f"\n✅ Processing completed! Generated {len(result_files)} anonymized images")
        print(f"Results saved in: {final_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CIAGAN one-click image processing')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Ensure running in correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = process_images_pipeline(args.input, args.output)
    
    if success:
        print("\n🎉 All steps completed! Your images have been successfully anonymized.")
    else:
        print("\n❌ Processing failed, please check error messages.")
        sys.exit(1)

if __name__ == "__main__":
    main()
