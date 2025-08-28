#!/usr/bin/env python3
"""
专门处理 high_score_images 目录的图片右下角高斯模糊脚本
"""

import cv2
import numpy as np
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_high_score_images():
    """
    处理 /home/zhiqics/sanjian/predata/output_frames70/high_score_images 目录中的所有图片
    对右下角进行高斯模糊处理
    """
    # 输入和输出目录
    input_dir = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images"
    output_dir = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images_blurred"
    
    # 模糊参数
    blur_ratio = 0.25      # 右下角1/4区域
    blur_strength = 121    # 模糊强度(必须为奇数) - 超强模糊效果
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 检查输入目录
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"在目录中没有找到图片文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图片文件")
    logger.info(f"开始处理右下角模糊，模糊区域比例: {blur_ratio}, 模糊强度: {blur_strength}")
    
    success_count = 0
    failed_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            # 读取图片
            image = cv2.imread(str(image_file))
            if image is None:
                logger.error(f"无法读取图片: {image_file.name}")
                failed_count += 1
                continue
            
            height, width = image.shape[:2]
            
            # 计算右下角模糊区域的位置
            start_x = int(width * (1 - blur_ratio))
            start_y = int(height * (1 - blur_ratio))
            
            # 提取右下角区域
            bottom_right_region = image[start_y:height, start_x:width]
            
            # 应用高斯模糊
            blurred_region = cv2.GaussianBlur(bottom_right_region, (blur_strength, blur_strength), 0)
            
            # 将模糊后的区域替换回原图
            result_image = image.copy()
            result_image[start_y:height, start_x:width] = blurred_region
            
            # 保存结果
            output_file_path = output_path / image_file.name
            cv2.imwrite(str(output_file_path), result_image)
            
            success_count += 1
            if i % 50 == 0:  # 每处理50张图片显示一次进度
                logger.info(f"已处理 {i}/{len(image_files)} 张图片...")
            
        except Exception as e:
            logger.error(f"处理图片 {image_file.name} 时出错: {str(e)}")
            failed_count += 1
    
    logger.info(f"处理完成! 成功: {success_count}, 失败: {failed_count}")
    logger.info(f"处理后的图片保存在: {output_dir}")

if __name__ == "__main__":
    process_high_score_images()
