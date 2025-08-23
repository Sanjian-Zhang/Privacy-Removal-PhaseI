#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量匿名化处理脚本 - CIAGAN
"""

import os
import sys
import shutil
import argparse
from tqdm import tqdm

def batch_anonymize_images(input_dir, output_dir):
    """
    批量匿名化处理图片
    """
    print("=== CIAGAN 批量匿名化处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 统计图片数量
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    if total_images == 0:
        print("❌ 输入目录中没有找到图片文件")
        return False
    
    print(f"发现 {total_images} 张图片需要处理")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 临时处理目录
    temp_dir = "/home/zhiqics/sanjian/baseline/ciagan/temp_batch_process"
    processed_dir = "/home/zhiqics/sanjian/baseline/ciagan/temp_batch_processed"
    
    try:
        # 步骤1: 数据预处理
        print("\n步骤1: 预处理图片数据...")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 添加源码路径
        source_path = os.path.join(os.path.dirname(__file__), 'source')
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        
        from process_data import get_lndm
        
        # 创建身份目录结构
        identity_dir = os.path.join(temp_dir, "0")
        os.makedirs(identity_dir, exist_ok=True)
        
        # 复制并重命名图片
        print("复制和重命名图片...")
        for i, filename in enumerate(tqdm(sorted(image_files))):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(identity_dir, f"{i:06d}.jpg")
            shutil.copy2(src_path, dst_path)
        
        # 处理人脸关键点
        print("处理人脸关键点和遮罩...")
        dlib_path = "source/"
        get_lndm(temp_dir, processed_dir, start_id=0, dlib_path=dlib_path)
        
        # 步骤2: 匿名化处理
        print("\n步骤2: 运行匿名化处理...")
        
        # 运行推理
        model_path = "/home/zhiqics/sanjian/baseline/ciagan/pretrained_models/modelG"
        temp_output = "/home/zhiqics/sanjian/baseline/ciagan/temp_anonymized"
        
        from test import run_inference
        
        run_inference(
            data_path=processed_dir,
            num_folders=1,
            model_path=model_path,
            output_path=temp_output
        )
        
        # 步骤3: 复制结果到最终输出目录
        print("\n步骤3: 复制结果到输出目录...")
        if os.path.exists(temp_output):
            anonymized_files = [f for f in os.listdir(temp_output) 
                              if f.lower().endswith('.jpg')]
            
            for i, anon_file in enumerate(tqdm(sorted(anonymized_files))):
                src_path = os.path.join(temp_output, anon_file)
                # 使用原始文件名
                original_name = image_files[i] if i < len(image_files) else f"anonymized_{i:06d}.jpg"
                dst_path = os.path.join(output_dir, f"anon_{original_name}")
                shutil.copy2(src_path, dst_path)
            
            print(f"✅ 成功处理 {len(anonymized_files)} 张图片")
            print(f"匿名化结果保存在: {output_dir}")
            
            # 清理临时文件
            print("清理临时文件...")
            for temp_folder in [temp_dir, processed_dir, temp_output]:
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
            
            return True
        else:
            print("❌ 匿名化处理失败，未找到输出文件")
            return False
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        for temp_folder in [temp_dir, processed_dir]:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
        
        return False

def main():
    parser = argparse.ArgumentParser(description='CIAGAN 批量匿名化处理')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    
    args = parser.parse_args()
    
    success = batch_anonymize_images(args.input, args.output)
    
    if success:
        print("\n🎉 批量匿名化处理完成!")
    else:
        print("\n❌ 批量匿名化处理失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
