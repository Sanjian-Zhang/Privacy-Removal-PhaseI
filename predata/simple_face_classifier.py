#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸数量分类器 - 简化版本
使用OpenCV进行人脸检测，按检测结果分类图片
"""

import os
import shutil
import cv2
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_faces_in_image(image_path, cascade_path=None):
    """检测图片中的人脸数量"""
    try:
        if cascade_path is None:
            cascade_path = '/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml'
        
        # 加载分类器
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("无法加载人脸检测器")
            return 0
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return 0
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return len(faces)
        
    except Exception as e:
        logger.error(f"检测图片 {image_path} 时出错: {e}")
        return 0


def create_directories(output_dir, max_faces=10):
    """创建分类目录"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i in range(max_faces + 1):
        face_dir = output_path / f"{i}_faces"
        face_dir.mkdir(exist_ok=True)
    
    many_faces_dir = output_path / "many_faces"
    many_faces_dir.mkdir(exist_ok=True)


def classify_images(input_dir, output_dir, max_faces=10, copy_mode=False):
    """分类图片"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    create_directories(output_dir, max_faces)
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 获取所有图片文件
    image_files = []
    for fmt in supported_formats:
        image_files.extend(input_path.glob(f"*{fmt}"))
        image_files.extend(input_path.glob(f"*{fmt.upper()}"))
    
    logger.info(f"找到 {len(image_files)} 个图片文件")
    
    # 统计信息
    stats = {}
    
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"处理 {i}/{len(image_files)}: {image_file.name}")
        
        # 检测人脸数量
        face_count = detect_faces_in_image(str(image_file))
        
        # 确定目标文件夹
        if face_count <= max_faces:
            target_dir = Path(output_dir) / f"{face_count}_faces"
        else:
            target_dir = Path(output_dir) / "many_faces"
        
        # 移动或复制文件
        try:
            target_file = target_dir / image_file.name
            
            # 处理重名文件
            counter = 1
            original_target = target_file
            while target_file.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_file = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            if copy_mode:
                shutil.copy2(str(image_file), str(target_file))
                action = "复制"
            else:
                shutil.move(str(image_file), str(target_file))
                action = "移动"
            
            logger.info(f"{action} {image_file.name} (人脸数: {face_count}) 到 {target_dir.name}")
            
            # 更新统计
            dir_name = target_dir.name
            stats[dir_name] = stats.get(dir_name, 0) + 1
            
        except Exception as e:
            logger.error(f"处理文件 {image_file.name} 时出错: {e}")
    
    # 打印统计信息
    logger.info("\n=== 分类统计 ===")
    for dir_name, count in sorted(stats.items()):
        logger.info(f"{dir_name}: {count} 张图片")
    logger.info(f"总共处理: {len(image_files)} 张图片")


def main():
    parser = argparse.ArgumentParser(description='按人脸数量分类图片')
    parser.add_argument('--input', '-i', 
                       default='/home/zhiqics/sanjian/predata/test_images',
                       help='输入图片目录')
    parser.add_argument('--output', '-o', 
                       default='/home/zhiqics/sanjian/predata/classified_by_faces',
                       help='输出目录')
    parser.add_argument('--max-faces', type=int, default=10,
                       help='最大人脸数分类')
    parser.add_argument('--copy', action='store_true',
                       help='复制模式（而不是移动）')
    
    args = parser.parse_args()
    
    try:
        classify_images(args.input, args.output, args.max_faces, args.copy)
        logger.info("分类完成!")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
