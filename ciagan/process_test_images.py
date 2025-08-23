#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process test images for CIAGAN
"""

import os
import shutil
import argparse
from source.process_data import get_lndm

def prepare_directory_structure(input_dir, temp_dir):
    """
    为 CIAGAN 数据处理准备目录结构
    CIAGAN 期望的目录结构：input_dir/identity_folder/image.jpg
    """
    # 创建临时目录结构
    os.makedirs(temp_dir, exist_ok=True)
    identity_dir = os.path.join(temp_dir, "identity_0")
    os.makedirs(identity_dir, exist_ok=True)
    
    # 复制所有图片到 identity_0 文件夹，重命名为数字
    counter = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(input_dir, filename)
            # 重命名为数字格式
            new_filename = f"{counter}.jpg"
            dst_path = os.path.join(identity_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} -> {new_filename} to {identity_dir}")
            counter += 1
    
    return temp_dir

def main():
    parser = argparse.ArgumentParser(description='Process images using CIAGAN pipeline')
    parser.add_argument('--input', type=str, 
                       default='/home/zhiqics/sanjian/dataset/test_images/images',
                       help='Input directory containing images')
    parser.add_argument('--output', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/processed_output/',
                       help='Output directory for processed data')
    parser.add_argument('--temp', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/temp_input/',
                       help='Temporary directory for processing')
    
    args = parser.parse_args()
    
    print("Processing test images with CIAGAN...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Temp directory: {args.temp}")
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist!")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 准备目录结构
    print("Preparing directory structure...")
    temp_input_dir = prepare_directory_structure(args.input, args.temp)
    
    # 使用 dlib 处理数据
    dlib_predictor_path = "source/"  # shape_predictor_68_face_landmarks.dat 在这个目录
    
    print("Processing images with dlib face landmarks...")
    try:
        # 确保输出目录存在
        for subdir in ['msk', 'clr', 'lndm', 'orig']:
            os.makedirs(os.path.join(args.output, subdir), exist_ok=True)
        
        get_lndm(temp_input_dir, args.output, start_id=0, dlib_path=dlib_predictor_path)
        print("Processing completed successfully!")
        print(f"Processed data saved to: {args.output}")
        
        # 列出生成的文件
        for root, dirs, files in os.walk(args.output):
            if files:
                print(f"\nGenerated files in {root}:")
                for file in files[:5]:  # 显示前5个文件
                    print(f"  - {file}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more files")
                    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    # 清理临时目录
    if os.path.exists(args.temp):
        shutil.rmtree(args.temp)
        print(f"Cleaned up temporary directory: {args.temp}")

if __name__ == "__main__":
    main()
