#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob
import sys
from pathlib import Path

def process_single_image_batch(input_dir, batch_files, batch_idx):
    
    print(f"\n=== Processing batch {batch_idx + 1} ({len(batch_files)} images) ===")
    
    temp_input = f"temp_input_{batch_idx}"
    temp_processed = f"temp_processed_{batch_idx}"
    temp_output = f"temp_output_{batch_idx}"
    
    try:
        print("1. Preparing input data...")
        identity_dir = os.path.join(temp_input, "identity_0")
        os.makedirs(identity_dir, exist_ok=True)
        
        for i, img_path in enumerate(batch_files):
            dst_path = os.path.join(identity_dir, f"{i}.jpg")
            shutil.copy2(img_path, dst_path)
        
        print("2. Running preprocessing...")
        cmd = f"python process_test_images.py --input {temp_input}/identity_0 --output {temp_processed} --temp temp_temp_{batch_idx}"
        result = os.system(cmd)
        
        if result != 0:
            print(f"❌ Preprocessing failed")
            return 0
        
        print("3. Fixing directory structure...")
        for subdir in ['clr', 'lndm', 'msk', 'orig']:
            old_path = os.path.join(temp_processed, subdir, "identity_0")
            new_path = os.path.join(temp_processed, subdir, "0")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                
                if os.path.exists(new_path):
                    files = sorted([f for f in os.listdir(new_path) if f.endswith('.jpg')])
                    for j, filename in enumerate(files):
                        old_file = os.path.join(new_path, filename)
                        new_file = os.path.join(new_path, f"{j:06d}.jpg")
                        if old_file != new_file:
                            os.rename(old_file, new_file)
        
        print("4. Running anonymization...")
        
        os.makedirs(temp_output, exist_ok=True)
        
        cmd = f"python source/test.py --data {temp_processed}/ --model pretrained_models/modelG --out {temp_output}/ --ids 1"
        result = os.system(cmd)
        
        if result != 0:
            print(f"❌ Anonymization failed")
            return 0
        
        result_files = glob.glob(os.path.join(temp_output, "*.jpg"))
        print(f"✅ Batch completed, generated {len(result_files)} images")
        
        return len(result_files)
        
    except Exception as e:
        print(f"❌ Batch processing exception: {e}")
        return 0
    
    finally:
        for temp_dir in [temp_input, temp_processed, temp_output, f"temp_temp_{batch_idx}"]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    input_dir = "/home/zhiqics/sanjian/dataset/images/Train"
    output_dir = "/home/zhiqics/sanjian/dataset/images/Train_anonymized/ciagan"
    
    print("=== CIAGAN Simplified Processing Script ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    total_images = len(image_files)
    
    if total_images == 0:
        print("❌ No image files found")
        return
    
    print(f"✅ Found {total_images} images")
    
    final_output = os.path.join(output_dir, "ciagan_anonymized")
    os.makedirs(final_output, exist_ok=True)
    
    batch_size = 3
    num_batches = (total_images + batch_size - 1) // batch_size
    print(f"Will process in {num_batches} batches, {batch_size} images per batch")
    
    successful_total = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        batch_success = process_single_image_batch(input_dir, batch_files, batch_idx)
        
        if batch_success > 0:
            temp_output = f"temp_output_{batch_idx}"
            if os.path.exists(temp_output):
                result_files = glob.glob(os.path.join(temp_output, "*.jpg"))
                for i, result_file in enumerate(result_files):
                    if i < len(batch_files):
                        original_name = os.path.basename(batch_files[i])
                        output_name = f"ciagan_{original_name}"
                    else:
                        output_name = f"ciagan_batch_{batch_idx}_{i:04d}.jpg"
                    
                    final_path = os.path.join(final_output, output_name)
                    shutil.copy2(result_file, final_path)
                
                successful_total += len(result_files)
                
                shutil.rmtree(temp_output, ignore_errors=True)
    
    print(f"\n🎉 Processing completed!")
    print(f"Successfully processed: {successful_total}/{total_images} images")
    print(f"Results saved in: {final_output}")

if __name__ == "__main__":
    main()
