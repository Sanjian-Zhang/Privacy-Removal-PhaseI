#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIAGAN 简化处理脚本 - 解决环境问题
"""

import os
import shutil
import glob
import sys
from pathlib import Path

def process_single_image_batch(input_dir, batch_files, batch_idx):
    """处理单个批次的图片"""
    
    print(f"\n=== 处理批次 {batch_idx + 1} ({len(batch_files)} 张图片) ===")
    
    # 临时目录
    temp_input = f"temp_input_{batch_idx}"
    temp_processed = f"temp_processed_{batch_idx}"
    temp_output = f"temp_output_{batch_idx}"
    
    try:
        # 1. 准备输入数据
        print("1. 准备输入数据...")
        identity_dir = os.path.join(temp_input, "identity_0")
        os.makedirs(identity_dir, exist_ok=True)
        
        # 复制图片并重命名
        for i, img_path in enumerate(batch_files):
            dst_path = os.path.join(identity_dir, f"{i}.jpg")
            shutil.copy2(img_path, dst_path)
        
        # 2. 运行预处理
        print("2. 运行预处理...")
        cmd = f"python process_test_images.py --input {temp_input}/identity_0 --output {temp_processed} --temp temp_temp_{batch_idx}"
        result = os.system(cmd)
        
        if result != 0:
            print(f"❌ 预处理失败")
            return 0
        
        # 3. 修复目录结构
        print("3. 修复目录结构...")
        for subdir in ['clr', 'lndm', 'msk', 'orig']:
            old_path = os.path.join(temp_processed, subdir, "identity_0")
            new_path = os.path.join(temp_processed, subdir, "0")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                
                # 重命名文件
                if os.path.exists(new_path):
                    files = sorted([f for f in os.listdir(new_path) if f.endswith('.jpg')])
                    for j, filename in enumerate(files):
                        old_file = os.path.join(new_path, filename)
                        new_file = os.path.join(new_path, f"{j:06d}.jpg")
                        if old_file != new_file:
                            os.rename(old_file, new_file)
        
        # 4. 运行匿名化
        print("4. 运行匿名化...")
        
        # 确保输出目录存在
        os.makedirs(temp_output, exist_ok=True)
        
        cmd = f"python source/test.py --data {temp_processed}/ --model pretrained_models/modelG --out {temp_output}/ --ids 1"
        result = os.system(cmd)
        
        if result != 0:
            print(f"❌ 匿名化失败")
            return 0
        
        # 5. 检查结果
        result_files = glob.glob(os.path.join(temp_output, "*.jpg"))
        print(f"✅ 批次完成，生成 {len(result_files)} 张图片")
        
        return len(result_files)
        
    except Exception as e:
        print(f"❌ 批次处理异常: {e}")
        return 0
    
    finally:
        # 清理临时文件
        for temp_dir in [temp_input, temp_processed, temp_output, f"temp_temp_{batch_idx}"]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    # 确保在正确目录
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # 输入输出路径
    input_dir = "/home/zhiqics/sanjian/dataset/images/Train"
    output_dir = "/home/zhiqics/sanjian/dataset/images/Train_anonymized/ciagan"
    
    print("=== CIAGAN 简化处理脚本 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 获取图片文件
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    total_images = len(image_files)
    
    if total_images == 0:
        print("❌ 没有找到图片文件")
        return
    
    print(f"✅ 找到 {total_images} 张图片")
    
    # 创建输出目录
    final_output = os.path.join(output_dir, "ciagan_anonymized")
    os.makedirs(final_output, exist_ok=True)
    
    # 分批处理 (小批次避免内存问题)
    batch_size = 3
    num_batches = (total_images + batch_size - 1) // batch_size
    print(f"将分 {num_batches} 批处理，每批 {batch_size} 张图片")
    
    successful_total = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        # 处理当前批次
        batch_success = process_single_image_batch(input_dir, batch_files, batch_idx)
        
        # 复制结果到最终输出目录
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
                
                # 清理临时输出
                shutil.rmtree(temp_output, ignore_errors=True)
    
    print(f"\n🎉 处理完成！")
    print(f"成功处理: {successful_total}/{total_images} 张图片")
    print(f"结果保存在: {final_output}")

if __name__ == "__main__":
    main()
