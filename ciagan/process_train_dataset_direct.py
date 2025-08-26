#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 CIAGAN 处理 Train 数据集的优化脚本 - 直接调用函数避免环境问题
"""

import os
import sys
import argparse
import shutil
import glob
import torch
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
    
    # 检查 dlib 模型
    dlib_model = script_dir / "source" / "shape_predictor_68_face_landmarks.dat"
    if not dlib_model.exists():
        print("❌ dlib 面部标记模型不存在")
        return False
    
    print("✅ 必要文件检查通过")
    return True

def process_single_batch_direct(image_files, batch_idx, batch_size, final_output):
    """
    直接处理单个批次，避免子进程调用
    """
    print(f"\n=== 处理第 {batch_idx + 1} 批 ({len(image_files)} 张图片) ===")
    
    try:
        # 导入必要的模块
        from source.process_data import get_lndm
        from source.test import main as test_main
        import source.test as test_module
        
        # 创建临时目录
        batch_temp_dir = f"temp_batch_{batch_idx}"
        batch_processed_dir = f"temp_batch_{batch_idx}_processed"
        batch_output_dir = f"temp_batch_{batch_idx}_output"
        
        # 步骤 1: 准备数据
        print("1. 准备批次数据...")
        batch_input_dir = os.path.join(batch_temp_dir, "identity_0")
        os.makedirs(batch_input_dir, exist_ok=True)
        
        # 复制并重命名图片
        for i, img_file in enumerate(image_files):
            dst_name = f"{i}.jpg"
            dst_path = os.path.join(batch_input_dir, dst_name)
            shutil.copy2(img_file, dst_path)
        
        # 步骤 2: 预处理 - 直接调用函数
        print("2. 预处理图片（提取面部特征）...")
        
        # 创建输出目录结构
        for subdir in ['msk', 'clr', 'lndm', 'orig']:
            os.makedirs(os.path.join(batch_processed_dir, subdir), exist_ok=True)
        
        # 调用 get_lndm 函数
        dlib_path = "source/"
        get_lndm(batch_temp_dir, batch_processed_dir, start_id=0, dlib_path=dlib_path)
        
        # 步骤 3: 修复目录结构
        print("3. 修复目录结构...")
        fix_directory_structure(batch_processed_dir)
        
        # 步骤 4: 运行匿名化 - 直接调用
        print("4. 运行身份匿名化...")
        
        # 模拟命令行参数
        sys.argv = [
            'test.py',
            '--data', batch_processed_dir + '/',
            '--model', 'pretrained_models/modelG',
            '--out', batch_output_dir + '/',
            '--ids', '1'
        ]
        
        # 调用测试函数
        test_main()
        
        # 步骤 5: 复制结果
        print("5. 复制结果...")
        if os.path.exists(batch_output_dir):
            result_files = glob.glob(os.path.join(batch_output_dir, "*.jpg"))
            successful_count = 0
            
            for i, result_file in enumerate(result_files):
                if i < len(image_files):
                    original_name = os.path.basename(image_files[i])
                    output_name = f"ciagan_{original_name}"
                else:
                    output_name = f"ciagan_batch_{batch_idx}_{i:04d}.jpg"
                
                output_path = os.path.join(final_output, output_name)
                shutil.copy2(result_file, output_path)
                successful_count += 1
            
            print(f"✅ 批次处理完成，生成 {successful_count} 张匿名化图片")
            return successful_count
        else:
            print("❌ 未找到输出文件")
            return 0
            
    except Exception as e:
        print(f"❌ 批次处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    finally:
        # 清理临时文件
        for temp_dir in [batch_temp_dir, batch_processed_dir, batch_output_dir]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

def fix_directory_structure(processed_dir):
    """修复处理后的目录结构"""
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

def process_train_images_direct(input_dir, output_dir, batch_size=10):
    """
    直接处理 Train 目录的图片，避免子进程调用
    """
    print("=== CIAGAN Train 数据集处理（直接调用版本）===")
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
        
        # 处理当前批次
        batch_success = process_single_batch_direct(batch_files, batch_idx, batch_size, final_output)
        successful_count += batch_success
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n🎉 所有批次处理完成！")
    print(f"成功处理: {successful_count}/{total_images} 张图片")
    print(f"结果保存在: {final_output}")
    
    return successful_count > 0

def main():
    parser = argparse.ArgumentParser(description='CIAGAN Train 数据集处理脚本（直接调用版本）')
    parser.add_argument('--input', type=str, 
                       default='/home/zhiqics/sanjian/dataset/images/Train',
                       help='输入图片目录')
    parser.add_argument('--output', type=str,
                       default='/home/zhiqics/sanjian/dataset/images/Train_anonymized/ciagan',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='批次大小 (默认: 5)')
    
    args = parser.parse_args()
    
    # 设置环境
    script_dir = setup_environment()
    
    # 检查必要条件
    if not check_prerequisites(script_dir):
        sys.exit(1)
    
    # 开始处理
    success = process_train_images_direct(args.input, args.output, args.batch_size)
    
    if success:
        print("\n🎉 Train 数据集匿名化处理完成！")
    else:
        print("\n❌ 处理失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
