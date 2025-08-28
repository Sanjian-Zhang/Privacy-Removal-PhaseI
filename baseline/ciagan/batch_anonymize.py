#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch anonymization processing script - CIAGAN
"""

import os
import sys
import shutil
import argparse
from tqdm import tqdm

def batch_anonymize_images(input_dir, output_dir):
    """
    Batch anonymization processing for images
    """
    print("=== CIAGAN Batch Anonymization Processing ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    if total_images == 0:
        print("❌ No image files found in input directory")
        return False
    
    print(f"Found {total_images} images to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporary processing directories
    temp_dir = "/home/zhiqics/sanjian/baseline/ciagan/temp_batch_process"
    processed_dir = "/home/zhiqics/sanjian/baseline/ciagan/temp_batch_processed"
    
    try:
        # Step 1: Data preprocessing
        print("\nStep 1: Preprocessing image data...")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 添加源码路径
        source_path = os.path.join(os.path.dirname(__file__), 'source')
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        
        from process_data import get_lndm
        
        # Create identity directory structure
        identity_dir = os.path.join(temp_dir, "0")
        os.makedirs(identity_dir, exist_ok=True)
        
        # Copy and rename images
        print("Copying and renaming images...")
        for i, filename in enumerate(tqdm(sorted(image_files))):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(identity_dir, f"{i:06d}.jpg")
            shutil.copy2(src_path, dst_path)
        
        # Process facial landmarks
        print("Processing facial landmarks and masks...")
        dlib_path = "source/"
        get_lndm(temp_dir, processed_dir, start_id=0, dlib_path=dlib_path)
        
        # Step 2: Anonymization processing
        print("\nStep 2: Running anonymization processing...")
        
        # Run inference
        model_path = "/home/zhiqics/sanjian/baseline/ciagan/pretrained_models/modelG"
        temp_output = "/home/zhiqics/sanjian/baseline/ciagan/temp_anonymized"
        
        from test import run_inference
        
        run_inference(
            data_path=processed_dir,
            num_folders=1,
            model_path=model_path,
            output_path=temp_output
        )
        
        # Step 3: Copy results to final output directory
        print("\nStep 3: Copying results to output directory...")
        if os.path.exists(temp_output):
            anonymized_files = [f for f in os.listdir(temp_output) 
                              if f.lower().endswith('.jpg')]
            
            for i, anon_file in enumerate(tqdm(sorted(anonymized_files))):
                src_path = os.path.join(temp_output, anon_file)
                # Use original filename
                original_name = image_files[i] if i < len(image_files) else f"anonymized_{i:06d}.jpg"
                dst_path = os.path.join(output_dir, f"anon_{original_name}")
                shutil.copy2(src_path, dst_path)
            
            print(f"✅ Successfully processed {len(anonymized_files)} images")
            print(f"Anonymized results saved in: {output_dir}")
            
            # Clean up temporary files
            print("Cleaning up temporary files...")
            for temp_folder in [temp_dir, processed_dir, temp_output]:
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
            
            return True
        else:
            print("❌ Anonymization processing failed, output files not found")
            return False
            
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary files
        for temp_folder in [temp_dir, processed_dir]:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
        
        return False

def main():
    parser = argparse.ArgumentParser(description='CIAGAN batch anonymization processing')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    success = batch_anonymize_images(args.input, args.output)
    
    if success:
        print("\n🎉 Batch anonymization processing completed!")
    else:
        print("\n❌ Batch anonymization processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
