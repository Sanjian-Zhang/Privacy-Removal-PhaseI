#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化批量匿名化处理脚本 - CIAGAN
处理大量图片并输出到指定目录
"""

import os
import sys
import shutil
import subprocess
import argparse
from math import ceil

def process_images_in_batches(input_dir, output_dir, batch_size=50):
    """
    分批处理图片，避免内存问题
    """
    print("=== CIAGAN 自动化批量匿名化处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {batch_size}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    total_images = len(image_files)
    if total_images == 0:
        print("❌ 输入目录中没有找到图片文件")
        return False
    
    print(f"发现 {total_images} 张图片需要处理")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算批次数量
    num_batches = ceil(total_images / batch_size)
    print(f"将分 {num_batches} 个批次处理")
    
    # 工作目录
    work_dir = "/home/zhiqics/sanjian/baseline/ciagan"
    
    successful_files = []
    
    for batch_idx in range(num_batches):
        print(f"\n=== 处理批次 {batch_idx + 1}/{num_batches} ===")
        
        # 计算当前批次的文件范围
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"处理文件 {start_idx + 1} 到 {end_idx} ({len(batch_files)} 张)")
        
        # 创建批次目录
        batch_input_dir = os.path.join(work_dir, f"batch_{batch_idx}")
        batch_processed_dir = os.path.join(work_dir, f"batch_{batch_idx}_processed")
        batch_output_dir = os.path.join(work_dir, f"batch_{batch_idx}_anonymized")
        
        try:
            # 步骤1: 复制当前批次的文件
            os.makedirs(batch_input_dir, exist_ok=True)
            for file in batch_files:
                src_path = os.path.join(input_dir, file)
                dst_path = os.path.join(batch_input_dir, file)
                shutil.copy2(src_path, dst_path)
            
            # 步骤2: 预处理数据
            print("  预处理数据...")
            cmd = [
                "python", "process_test_images.py",
                "--input", batch_input_dir,
                "--output", batch_processed_dir
            ]
            
            result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ 预处理失败: {result.stderr}")
                continue
            
            # 步骤3: 修复文件夹命名
            print("  修复文件夹结构...")
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                old_path = os.path.join(batch_processed_dir, subdir, 'identity_0')
                new_path = os.path.join(batch_processed_dir, subdir, '0')
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            
            # 步骤4: 修复文件名格式
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                subdir_path = os.path.join(batch_processed_dir, subdir, '0')
                if os.path.exists(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.jpg'):
                            old_file = os.path.join(subdir_path, file)
                            new_file = os.path.join(subdir_path, f"{int(file.split('.')[0]):06d}.jpg")
                            if old_file != new_file:
                                os.rename(old_file, new_file)
            
            # 步骤5: 运行匿名化
            print("  运行匿名化...")
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
                print(f"  ❌ 匿名化失败: {result.stderr}")
                continue
            
            # 步骤6: 复制结果到最终输出目录
            print("  复制结果...")
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
            
            print(f"  ✅ 批次 {batch_idx + 1} 完成，处理了 {len(batch_files)} 张图片")
            
        except Exception as e:
            print(f"  ❌ 批次 {batch_idx + 1} 处理失败: {e}")
        
        finally:
            # 清理临时文件
            for temp_dir in [batch_input_dir, batch_processed_dir, batch_output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {len(successful_files)}/{total_images} 张图片")
    print(f"输出目录: {output_dir}")
    
    return len(successful_files) > 0

def main():
    parser = argparse.ArgumentParser(description='CIAGAN 自动化批量匿名化处理')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='每批处理的图片数量 (默认: 50)')
    
    args = parser.parse_args()
    
    success = process_images_in_batches(args.input, args.output, args.batch_size)
    
    if success:
        print("\n🎉 批量匿名化处理完成!")
    else:
        print("\n❌ 批量匿名化处理失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
