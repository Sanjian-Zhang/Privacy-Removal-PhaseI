#!/usr/bin/env python3
"""
分批处理图片的脚本，避免一次性加载太多图片导致内存溢出
"""
import os
import sys
import glob
import subprocess
import time
from pathlib import Path

def process_batch(input_dir, output_dir, batch_size=5, model="fdf128_rcnn512"):
    """
    分批处理图片
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        batch_size: 每批处理的图片数量
        model: 使用的模型
    """
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_dir, ext)))
        all_images.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    all_images.sort()
    total_images = len(all_images)
    
    print(f"找到 {total_images} 张图片")
    print(f"将分 {(total_images + batch_size - 1) // batch_size} 批处理，每批 {batch_size} 张")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 分批处理
    for i in range(0, total_images, batch_size):
        batch_num = i // batch_size + 1
        batch_images = all_images[i:i + batch_size]
        
        print(f"\n开始处理第 {batch_num} 批 ({len(batch_images)} 张图片):")
        for img in batch_images:
            print(f"  - {os.path.basename(img)}")
        
        # 为每张图片单独处理
        for img_path in batch_images:
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"anonymized_{img_name}")
            
            # 检查是否已经处理过
            if os.path.exists(output_path):
                print(f"  跳过已处理的图片: {img_name}")
                continue
            
            print(f"  正在处理: {img_name}")
            
            try:
                # 构建 DeepPrivacy 命令
                cmd = [
                    "python", "anonymize.py",
                    "-s", img_path,
                    "-t", output_path,
                    "-m", model
                ]
                
                # 运行处理命令
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"  ✅ 成功处理: {img_name}")
                else:
                    print(f"  ❌ 处理失败: {img_name}")
                    print(f"     错误: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ 处理超时: {img_name}")
            except Exception as e:
                print(f"  ❌ 处理异常: {img_name} - {str(e)}")
        
        print(f"第 {batch_num} 批处理完成")
        
        # 简短休息以释放内存
        if i + batch_size < total_images:
            print("等待 2 秒释放内存...")
            time.sleep(2)
    
    print(f"\n所有批次处理完成！")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python batch_process.py <输入目录> <输出目录> [批次大小] [模型名称]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    model = sys.argv[4] if len(sys.argv) > 4 else "fdf128_rcnn512"
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    process_batch(input_dir, output_dir, batch_size, model)
