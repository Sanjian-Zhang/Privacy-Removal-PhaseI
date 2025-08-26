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
    完整的图片处理流水线
    """
    print("=== CIAGAN 图片处理流水线 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 检查图片文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"❌ 输入目录中没有找到图片文件")
        return False
    
    print(f"✅ 找到 {len(image_files)} 张图片")
    
    # 创建临时目录
    temp_processed = os.path.join(output_dir, "temp_processed")
    final_output = os.path.join(output_dir, "anonymized")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    try:
        # 步骤 1: 预处理图片
        print("\n=== 步骤 1: 预处理图片 ===")
        cmd1 = [
            "python", "process_test_images.py",
            "--input", input_dir,
            "--output", temp_processed
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            print(f"❌ 预处理失败: {result1.stderr}")
            return False
        
        print("✅ 图片预处理完成")
        
        # 步骤 2: 修复文件名格式
        print("\n=== 步骤 2: 修复文件名格式 ===")
        identity_dir = os.path.join(temp_processed, "clr", "identity_0")
        if os.path.exists(identity_dir):
            # 重命名 identity_0 为 0
            new_identity_dir = os.path.join(temp_processed, "clr", "0")
            os.rename(identity_dir, new_identity_dir)
            
            # 对所有子目录执行相同操作
            for subdir in ['lndm', 'msk', 'orig']:
                old_path = os.path.join(temp_processed, subdir, "identity_0")
                new_path = os.path.join(temp_processed, subdir, "0")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            
            # 重命名文件为 6 位数格式
            for subdir in ['clr', 'lndm', 'msk', 'orig']:
                subdir_path = os.path.join(temp_processed, subdir, "0")
                if os.path.exists(subdir_path):
                    files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                    for i, filename in enumerate(files):
                        old_file = os.path.join(subdir_path, filename)
                        new_file = os.path.join(subdir_path, f"{i:06d}.jpg")
                        os.rename(old_file, new_file)
        
        print("✅ 文件名格式修复完成")
        
        # 步骤 3: 运行身份匿名化
        print("\n=== 步骤 3: 运行身份匿名化 ===")
        cmd2 = [
            "python", "source/test.py",
            "--data", temp_processed + "/",
            "--model", "pretrained_models/modelG",
            "--out", final_output + "/",
            "--ids", "1"
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"❌ 身份匿名化失败: {result2.stderr}")
            return False
        
        print("✅ 身份匿名化完成")
        
        # 清理临时文件
        if os.path.exists(temp_processed):
            shutil.rmtree(temp_processed)
        
        # 检查结果
        result_files = [f for f in os.listdir(final_output) if f.endswith('.jpg')]
        print(f"\n✅ 处理完成！生成了 {len(result_files)} 张匿名化图片")
        print(f"结果保存在: {final_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CIAGAN 一键图片处理')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 确保在正确的目录中运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = process_images_pipeline(args.input, args.output)
    
    if success:
        print("\n🎉 所有步骤完成！您的图片已成功匿名化。")
    else:
        print("\n❌ 处理失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
