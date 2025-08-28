#!/usr/bin/env python3
"""
DeepPrivacy批量图片匿名化脚本
"""
import os
import sys
import argparse
from pathlib import Path

# 添加DeepPrivacy到Python路径
sys.path.insert(0, '/home/zhiqics/sanjian/baseline/DeepPrivacy')

def batch_anonymize(input_dir, output_dir, model='fdf128_rcnn512'):
    """
    批量匿名化图片
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        model: 使用的模型名称
    """
    import glob
    from deep_privacy import build_anonymizer
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建匿名器
    print(f"Loading model: {model}")
    try:
        anonymizer = build_anonymizer(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 获取所有图片文件
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(input_dir, ext)))
        img_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(img_files)} images to process")
    
    # 处理每个图片
    for i, img_path in enumerate(img_files):
        try:
            print(f"Processing {i+1}/{len(img_files)}: {os.path.basename(img_path)}")
            
            # 获取输出文件名
            img_name = os.path.basename(img_path)
            name, ext = os.path.splitext(img_name)
            output_path = os.path.join(output_dir, f"{name}_anonymized{ext}")
            
            # 匿名化
            from pathlib import Path
            anonymizer.anonymize_image_paths([Path(img_path)], [Path(output_path)])
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print("Batch processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Batch anonymize images using DeepPrivacy')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for anonymized images')
    parser.add_argument('--model', '-m', default='fdf128_rcnn512', 
                       choices=['deep_privacy_v1', 'fdf128_rcnn512', 'fdf128_retinanet512', 
                               'fdf128_retinanet256', 'fdf128_retinanet128'],
                       help='Model to use for anonymization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return
    
    batch_anonymize(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
