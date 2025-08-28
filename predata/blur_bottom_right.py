#!/usr/bin/env python3
"""
图片右下角高斯模糊处理脚本
对指定目录中的图片右下角区域进行高斯模糊处理
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_gaussian_blur_bottom_right(image_path, output_path, blur_ratio=0.25, blur_strength=15):
    """
    对图片右下角区域应用高斯模糊
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        blur_ratio: 模糊区域占图片的比例 (0.25表示右下角1/4区域)
        blur_strength: 模糊强度 (高斯核大小)
    """
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return False
        
        height, width = image.shape[:2]
        
        # 计算右下角模糊区域的位置
        # 模糊区域从图片的 (1-blur_ratio) 位置开始到右下角
        start_x = int(width * (1 - blur_ratio))
        start_y = int(height * (1 - blur_ratio))
        
        # 提取右下角区域
        bottom_right_region = image[start_y:height, start_x:width]
        
        # 确保blur_strength为奇数
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # 应用高斯模糊
        blurred_region = cv2.GaussianBlur(bottom_right_region, (blur_strength, blur_strength), 0)
        
        # 将模糊后的区域替换回原图
        result_image = image.copy()
        result_image[start_y:height, start_x:width] = blurred_region
        
        # 保存结果
        cv2.imwrite(output_path, result_image)
        logger.info(f"已处理: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
        return False

def process_directory(input_dir, output_dir, blur_ratio=0.0625, blur_strength=15, overwrite=False):
    """
    批量处理目录中的所有图片
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        blur_ratio: 模糊区域占图片的比例
        blur_strength: 模糊强度
        overwrite: 是否覆盖原文件
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 设置输出路径
    if overwrite:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"在目录 {input_dir} 中没有找到图片文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    success_count = 0
    failed_count = 0
    
    for image_file in image_files:
        if overwrite:
            # 覆盖原文件
            output_file_path = str(image_file)
        else:
            # 保存到输出目录
            output_file_path = str(output_path / image_file.name)
        
        if apply_gaussian_blur_bottom_right(str(image_file), output_file_path, blur_ratio, blur_strength):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info(f"处理完成! 成功: {success_count}, 失败: {failed_count}")

def main():
    parser = argparse.ArgumentParser(description='对图片右下角进行高斯模糊处理')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('-o', '--output', help='输出目录路径 (如果不指定则覆盖原文件)')
    parser.add_argument('-r', '--ratio', type=float, default=0.25, 
                       help='模糊区域比例 (默认: 0.25, 即右下角1/4区域)')
    parser.add_argument('-s', '--strength', type=int, default=15, 
                       help='模糊强度 (默认: 15)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='覆盖原文件 (如果指定则忽略输出目录)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.ratio <= 0 or args.ratio > 1:
        logger.error("模糊区域比例必须在 0 到 1 之间")
        return
    
    if args.strength <= 0:
        logger.error("模糊强度必须大于 0")
        return
    
    # 确定输出目录
    if args.overwrite or not args.output:
        overwrite = True
        output_dir = args.input_dir
    else:
        overwrite = False
        output_dir = args.output
    
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模糊区域比例: {args.ratio}")
    logger.info(f"模糊强度: {args.strength}")
    logger.info(f"覆盖原文件: {overwrite}")
    
    # 开始处理
    process_directory(args.input_dir, output_dir, args.ratio, args.strength, overwrite)

if __name__ == "__main__":
    main()
