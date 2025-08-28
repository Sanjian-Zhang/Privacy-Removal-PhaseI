#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 CIAGAN 处理 Train 数据集的专用脚本
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob
from pathlib import Path

def setup_environment():
    """设置环境"""
    print("🔧 设置 CIAGAN 环境...")
    
    # 确保在 CIAGAN 目录中
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # 添加源码路径
    source_path = script_dir / "source"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    
    print(f"✅ 工作目录: {os.getcwd()}")
    return script_dir

def check_prerequisites(script_dir):
    """检查必要文件和目录"""
    print("🔍 检查必要文件...")
    
    # 检查预训练模型
    model_path = script_dir / "pretrained_models" / "modelG.pth"
    if not model_path.exists():
        print("❌ 预训练模型不存在，请先运行: python setup_model.py")
        return False
    
    # 检查源码文件
    test_script = script_dir / "source" / "test.py"
    if not test_script.exists():
        print("❌ 源码文件不完整")
        return False
    
    print("✅ 必要文件检查通过")
    return True

def process_train_images(input_dir, output_dir, batch_size=50):
    """
    处理 Train 目录的图片
    
    Args:
        input_dir: 输入目录 (/home/zhiqics/sanjian/dataset/images/Train)
        output_dir: 输出目录
        batch_size: 批次大小，避免内存不足
    """
    print("=== CIAGAN Train 数据集处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.png")))
    image_files = sorted(image_files)
    
    if not image_files:
        print("❌ 输入目录中没有找到图片文件")
        return False
    
    total_images = len(image_files)
    print(f"✅ 找到 {total_images} 张图片")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    final_output = os.path.join(output_dir, "ciagan_anonymized")
    os.makedirs(final_output, exist_ok=True)
    
    # 分批处理
    num_batches = (total_images + batch_size - 1) // batch_size
    print(f"将分 {num_batches} 批处理，每批最多 {batch_size} 张图片")
    
    successful_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"\n=== 处理第 {batch_idx + 1}/{num_batches} 批 ({len(batch_files)} 张图片) ===")
        
        # 创建临时批次目录
        batch_temp_dir = f"temp_batch_{batch_idx}"
        batch_processed_dir = f"temp_batch_{batch_idx}_processed"
        batch_output_dir = f"temp_batch_{batch_idx}_output"
        
        try:
            # 步骤 1: 准备批次数据
            print("1. 准备批次数据...")
            batch_input_dir = os.path.join(batch_temp_dir, "input")
            os.makedirs(batch_input_dir, exist_ok=True)
            
            # 复制批次图片
            for i, img_file in enumerate(batch_files):
                dst_name = f"batch_{batch_idx}_{i:04d}.jpg"
                dst_path = os.path.join(batch_input_dir, dst_name)
                shutil.copy2(img_file, dst_path)
            
            # 步骤 2: 预处理
            print("2. 预处理图片...")
            
            # 确保环境变量正确传递
            env = os.environ.copy()
            env['CONDA_DEFAULT_ENV'] = 'ciagan'
            
            cmd_preprocess = [
                "python", "process_test_images.py",
                "--input", batch_input_dir,
                "--output", batch_processed_dir
            ]
            
            result1 = subprocess.run(cmd_preprocess, capture_output=True, text=True, env=env)
            if result1.returncode != 0:
                print(f"❌ 批次 {batch_idx + 1} 预处理失败: {result1.stderr}")
                continue
            
            # 步骤 3: 修复目录结构
            print("3. 修复目录结构...")
            fix_directory_structure(batch_processed_dir)
            
            # 步骤 4: 运行匿名化
            print("4. 运行身份匿名化...")
            cmd_anonymize = [
                "python", "source/test.py",
                "--data", batch_processed_dir + "/",
                "--model", "pretrained_models/modelG",
                "--out", batch_output_dir + "/",
                "--ids", "1"
            ]
            
            result2 = subprocess.run(cmd_anonymize, capture_output=True, text=True, env=env)
            if result2.returncode != 0:
                print(f"❌ 批次 {batch_idx + 1} 匿名化失败: {result2.stderr}")
                continue
            
            # 步骤 5: 复制结果
            print("5. 复制结果...")
            if os.path.exists(batch_output_dir):
                result_files = glob.glob(os.path.join(batch_output_dir, "*.jpg"))
                for i, result_file in enumerate(result_files):
                    # 生成输出文件名，保持与原始文件的对应关系
                    if i < len(batch_files):
                        original_name = os.path.basename(batch_files[i])
                        output_name = f"ciagan_{original_name}"
                    else:
                        output_name = f"ciagan_batch_{batch_idx}_{i:04d}.jpg"
                    
                    output_path = os.path.join(final_output, output_name)
                    shutil.copy2(result_file, output_path)
                
                successful_count += len(result_files)
                print(f"✅ 批次 {batch_idx + 1} 完成，生成 {len(result_files)} 张匿名化图片")
            
        except Exception as e:
            print(f"❌ 批次 {batch_idx + 1} 处理失败: {e}")
        
        finally:
            # 清理临时文件
            for temp_dir in [batch_temp_dir, batch_processed_dir, batch_output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    print(f"\n🎉 所有批次处理完成！")
    print(f"成功处理: {successful_count}/{total_images} 张图片")
    print(f"结果保存在: {final_output}")
    
    return successful_count > 0

def fix_directory_structure(processed_dir):
    """修复处理后的目录结构"""
    # CIAGAN 期望的目录结构: processed_dir/[clr,lndm,msk,orig]/0/
    for subdir in ['clr', 'lndm', 'msk', 'orig']:
        identity_dir = os.path.join(processed_dir, subdir, "identity_0")
        target_dir = os.path.join(processed_dir, subdir, "0")
        
        if os.path.exists(identity_dir) and not os.path.exists(target_dir):
            os.rename(identity_dir, target_dir)
        
        # 重命名文件为6位数格式
        if os.path.exists(target_dir):
            files = sorted([f for f in os.listdir(target_dir) if f.endswith('.jpg')])
            for i, filename in enumerate(files):
                old_path = os.path.join(target_dir, filename)
                new_path = os.path.join(target_dir, f"{i:06d}.jpg")
                if old_path != new_path:
                    os.rename(old_path, new_path)

def main():
    parser = argparse.ArgumentParser(description='CIAGAN Train 数据集处理脚本')
    parser.add_argument('--input', type=str, 
                       default='/home/zhiqics/sanjian/dataset/images/Train',
                       help='输入图片目录')
    parser.add_argument('--output', type=str,
                       default='/home/zhiqics/sanjian/dataset/images/Train_anonymized/ciagan',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='批次大小 (默认: 20)')
    
    args = parser.parse_args()
    
    # 设置环境
    script_dir = setup_environment()
    
    # 检查必要条件
    if not check_prerequisites(script_dir):
        sys.exit(1)
    
    # 开始处理
    success = process_train_images(args.input, args.output, args.batch_size)
    
    if success:
        print("\n🎉 Train 数据集匿名化处理完成！")
    else:
        print("\n❌ 处理失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
