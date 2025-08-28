#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import argparse
from math import ceil

def process_images_in_batches(input_dir, output_dir, batch_size=50):
    print("=== CIAGAN Auto Batch Anonymization ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return False
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    total_images = len(image_files)
    if total_images == 0:
        print("❌ No image files found in input directory")
        return False
    
    print(f"Found {total_images} images to process")
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_batches = ceil(total_images / batch_size)
    print(f"Will process in {num_batches} batches")
    
    work_dir = "/home/zhiqics/sanjian/baseline/ciagan"
    
    successful_files = []
    
    for batch_idx in range(num_batches):
        print(f"\n=== Processing batch {batch_idx + 1}/{num_batches} ===")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"Processing files {start_idx + 1} to {end_idx} ({len(batch_files)} images)")
        
        batch_input_dir = os.path.join(work_dir, f"batch_{batch_idx}")
        batch_processed_dir = os.path.join(work_dir, f"batch_{batch_idx}_processed")
        batch_output_dir = os.path.join(work_dir, f"batch_{batch_idx}_anonymized")
        
        try:
            os.makedirs(batch_input_dir, exist_ok=True)
            for file in batch_files:
                src_path = os.path.join(input_dir, file)
                dst_path = os.path.join(batch_input_dir, file)
                shutil.copy2(src_path, dst_path)
            
            print("  Preprocessing data...")
            cmd = [
                "python", "process_test_images.py",
                "--input", batch_input_dir,
                "--output", batch_processed_dir
            ]
            
            result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ Preprocessing failed: {result.stderr}")
                continue
            
            print("  Fixing folder structure...")
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                old_path = os.path.join(batch_processed_dir, subdir, 'identity_0')
                new_path = os.path.join(batch_processed_dir, subdir, '0')
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                subdir_path = os.path.join(batch_processed_dir, subdir, '0')
                if os.path.exists(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.jpg'):
                            old_file = os.path.join(subdir_path, file)
                            new_file = os.path.join(subdir_path, f"{int(file.split('.')[0]):06d}.jpg")
                            if old_file != new_file:
                                os.rename(old_file, new_file)
            
            print("  Running anonymization...")
            os.makedirs(batch_output_dir, exist_ok=True)
            
            cmd = [
                "python", "source/test.py",
                "--data", batch_processed_dir + "/",
                "--model", "pretrained_models/modelG",
                "--out", batch_output_dir + "/",
                "--ids", "1"
            ]
            
            result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ Anonymization failed: {result.stderr}")
                continue
            
            print("  Copying results...")
            if os.path.exists(batch_output_dir):
                anonymized_files = sorted([f for f in os.listdir(batch_output_dir) 
                                         if f.lower().endswith('.jpg')])
                
                for i, anon_file in enumerate(anonymized_files):
                    if i < len(batch_files):
                        src_path = os.path.join(batch_output_dir, anon_file)
                        original_name = batch_files[i]
                        dst_path = os.path.join(output_dir, f"anon_{original_name}")
                        shutil.copy2(src_path, dst_path)
                        successful_files.append(original_name)
            
            print(f"  ✅ Batch {batch_idx + 1} completed, processed {len(batch_files)} images")
            
        except Exception as e:
            print(f"  ❌ Batch {batch_idx + 1} processing failed: {e}")
        
        finally:
            for temp_dir in [batch_input_dir, batch_processed_dir, batch_output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    print(f"\n=== Processing completed ===")
    print(f"Successfully processed: {len(successful_files)}/{total_images} images")
    print(f"Output directory: {output_dir}")
    
    return len(successful_files) > 0

def main():
    parser = argparse.ArgumentParser(description='CIAGAN Auto Batch Anonymization')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of images per batch (default: 50)')
    
    args = parser.parse_args()
    
    success = process_images_in_batches(args.input, args.output, args.batch_size)
    
    if success:
        print("\n🎉 Batch anonymization completed!")
    else:
        print("\n❌ Batch anonymization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
