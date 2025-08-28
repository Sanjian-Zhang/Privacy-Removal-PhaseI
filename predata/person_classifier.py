#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体数量分类器 - 使用YOLOv8模型
检测图片中的人体数量并进行分类
"""

import os
import shutil
import cv2
from pathlib import Path
import argparse
import logging

# 检查ultralytics是否可用
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics未安装，将只使用OpenCV检测")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_persons_yolo(image_path, model_path):
    """使用YOLO检测人体数量"""
    if not YOLO_AVAILABLE:
        logger.error("YOLO不可用，请安装ultralytics")
        return 0
        
    try:
        model = YOLO(model_path)
        results = model(image_path, conf=0.5)
        
        person_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # COCO数据集中，class 0 是person
                    if class_id == 0 and confidence >= 0.5:
                        person_count += 1
        
        return person_count
        
    except Exception as e:
        logger.error(f"YOLO检测失败: {e}")
        return 0


def detect_persons_opencv(image_path, cascade_path=None):
    """使用OpenCV检测人体/人脸数量"""
    try:
        if cascade_path is None:
            cascade_path = '/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml'
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("无法加载人脸检测器")
            return 0
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return 0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        return len(faces)
        
    except Exception as e:
        logger.error(f"OpenCV检测失败: {e}")
        return 0


def create_directories(output_dir, max_persons=10):
    """创建分类目录"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i in range(max_persons + 1):
        person_dir = output_path / f"{i}_persons"
        person_dir.mkdir(exist_ok=True)
    
    many_persons_dir = output_path / "many_persons"
    many_persons_dir.mkdir(exist_ok=True)


def classify_images(input_dir, output_dir, model_path=None, use_yolo=True, max_persons=10, copy_mode=False):
    """分类图片"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 检查模型
    if use_yolo and model_path and not Path(model_path).exists():
        logger.error(f"YOLO模型文件不存在: {model_path}")
        use_yolo = False
    
    if use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO不可用，切换到OpenCV模式")
        use_yolo = False
    
    # 创建输出目录
    create_directories(output_dir, max_persons)
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 获取所有图片文件
    image_files = []
    for fmt in supported_formats:
        image_files.extend(input_path.glob(f"*{fmt}"))
        image_files.extend(input_path.glob(f"*{fmt.upper()}"))
    
    logger.info(f"找到 {len(image_files)} 个图片文件")
    logger.info(f"使用检测方法: {'YOLO' if use_yolo else 'OpenCV'}")
    
    # 统计信息
    stats = {}
    
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"处理 {i}/{len(image_files)}: {image_file.name}")
        
        # 检测人体/人脸数量
        if use_yolo:
            person_count = detect_persons_yolo(str(image_file), model_path)
        else:
            person_count = detect_persons_opencv(str(image_file))
        
        # 确定目标文件夹
        if person_count <= max_persons:
            target_dir = Path(output_dir) / f"{person_count}_persons"
        else:
            target_dir = Path(output_dir) / "many_persons"
        
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
            
            method = "YOLO" if use_yolo else "OpenCV"
            logger.info(f"{action} {image_file.name} ({method}检测: {person_count}) 到 {target_dir.name}")
            
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
    parser = argparse.ArgumentParser(description='按人体/人脸数量分类图片')
    parser.add_argument('--input', '-i', 
                       default='/home/zhiqics/sanjian/predata/test_images',
                       help='输入图片目录')
    parser.add_argument('--output', '-o', 
                       default='/home/zhiqics/sanjian/predata/classified_by_persons',
                       help='输出目录')
    parser.add_argument('--model', '-m',
                       default='/home/zhiqics/sanjian/predata/models/yolov8s.pt',
                       help='YOLO模型文件路径')
    parser.add_argument('--max-persons', type=int, default=10,
                       help='最大人数分类')
    parser.add_argument('--copy', action='store_true',
                       help='复制模式（而不是移动）')
    parser.add_argument('--opencv', action='store_true',
                       help='强制使用OpenCV而不是YOLO')
    
    args = parser.parse_args()
    
    try:
        use_yolo = not args.opencv
        classify_images(
            args.input, 
            args.output, 
            args.model if use_yolo else None, 
            use_yolo, 
            args.max_persons, 
            args.copy
        )
        logger.info("分类完成!")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
