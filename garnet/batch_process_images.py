#!/usr/bin/env python3
"""
批量处理图片的GaRNet文本去除脚本
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import torch
import subprocess

# 添加CODE目录到路径
garnet_path = "/home/zhiqics/sanjian/baseline/garnet"
sys.path.append(os.path.join(garnet_path, "CODE"))

def setup_environment():
    """设置环境和激活虚拟环境"""
    print("Setting up GaRNet environment...")
    
    # 检查是否在garnet目录
    if not os.path.exists(os.path.join(garnet_path, "WEIGHTS/GaRNet/saved_model.pth")):
        print(f"Error: GaRNet model not found at {garnet_path}")
        print("Please make sure GaRNet is properly set up.")
        return False
    
    return True

def detect_text_regions(image_path):
    """
    简单的文本区域检测
    这里使用OpenCV的EAST文本检测器或简单的边缘检测
    在实际应用中，你可能需要使用更高级的文本检测模型
    """
    print(f"Detecting text regions in {image_path}...")
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # 简单示例：创建一些虚拟的文本框
    # 在实际应用中，你需要使用真正的文本检测算法
    text_boxes = []
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小的区域（可能不是文本）
        if w > 30 and h > 10 and w < width * 0.8 and h < height * 0.3:
            # 将矩形转换为4个点的格式 (x1,y1,x2,y2,x3,y3,x4,y4)
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            
            text_boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    
    # 如果没有检测到文本，创建一个示例文本框用于演示
    if not text_boxes:
        # 在图片中心创建一个示例文本框
        center_x, center_y = width // 2, height // 2
        box_width, box_height = min(200, width // 3), min(50, height // 6)
        
        x1 = center_x - box_width // 2
        y1 = center_y - box_height // 2
        x2 = center_x + box_width // 2
        y2 = center_y - box_height // 2
        x3 = center_x + box_width // 2
        y3 = center_y + box_height // 2
        x4 = center_x - box_width // 2
        y4 = center_y + box_height // 2
        
        text_boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        print(f"No text detected, created demo box at center")
    
    print(f"Detected {len(text_boxes)} text regions")
    return text_boxes

def create_text_file(image_path, text_boxes, output_dir):
    """创建对应的文本框坐标文件"""
    image_name = Path(image_path).stem
    txt_file = os.path.join(output_dir, f"{image_name}.txt")
    
    with open(txt_file, 'w') as f:
        for box in text_boxes:
            # 格式: x1,y1,x2,y2,x3,y3,x4,y4
            line = ','.join(map(str, box))
            f.write(line + '\n')
    
    print(f"Created text file: {txt_file}")
    return txt_file

def process_images_with_garnet(input_dir, output_dir, temp_img_dir, temp_txt_dir):
    """使用GaRNet处理图片"""
    print("Processing images with GaRNet...")
    
    # 切换到garnet目录
    original_cwd = os.getcwd()
    os.chdir(garnet_path)
    
    try:
        # 激活虚拟环境并运行inference
        cmd = [
            "bash", "-c",
            f"source garnet_env/bin/activate && "
            f"cd CODE && "
            f"python inference.py --gpu "
            f"--image_path {temp_img_dir} "
            f"--box_path {temp_txt_dir} "
            f"--result_path {output_dir}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ GaRNet processing completed successfully")
            print("STDOUT:", result.stdout)
        else:
            print("✗ GaRNet processing failed")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        print(f"Error running GaRNet: {e}")
        return False
    finally:
        os.chdir(original_cwd)
    
    return True

def main():
    """主函数"""
    input_dir = "/home/zhiqics/sanjian/dataset/test_images/images"
    output_dir = "/home/zhiqics/sanjian/dataset/test_images/anon/garnet"
    
    print("="*60)
    print("GaRNet 批量文本去除处理")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查环境
    if not setup_environment():
        return False
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录存放处理的图片和文本文件
    temp_base = "/tmp/garnet_processing"
    temp_img_dir = os.path.join(temp_base, "images")
    temp_txt_dir = os.path.join(temp_base, "txt")
    
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_txt_dir, exist_ok=True)
    
    try:
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print("No image files found in input directory")
            return False
        
        print(f"Found {len(image_files)} images to process")
        
        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_file}...")
            
            input_image_path = os.path.join(input_dir, image_file)
            temp_image_path = os.path.join(temp_img_dir, image_file)
            
            # 复制图片到临时目录
            shutil.copy2(input_image_path, temp_image_path)
            
            # 检测文本区域
            text_boxes = detect_text_regions(input_image_path)
            
            # 创建文本坐标文件
            create_text_file(temp_image_path, text_boxes, temp_txt_dir)
        
        # 使用GaRNet批量处理
        if process_images_with_garnet(input_dir, output_dir, temp_img_dir, temp_txt_dir):
            print(f"\n✓ 所有图片处理完成!")
            print(f"✓ 结果保存在: {output_dir}")
            
            # 列出处理后的文件
            if os.path.exists(output_dir):
                processed_files = os.listdir(output_dir)
                print(f"✓ 生成了 {len(processed_files)} 个处理后的文件:")
                for file in processed_files:
                    print(f"  - {file}")
        else:
            print("✗ 处理失败")
            return False
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_base):
            shutil.rmtree(temp_base)
            print("Cleaned up temporary files")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 批量处理成功完成!")
    else:
        print("\n❌ 批量处理失败")
        sys.exit(1)
