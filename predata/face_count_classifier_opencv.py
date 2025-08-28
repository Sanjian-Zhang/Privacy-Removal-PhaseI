#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸数量分类器 - 使用OpenCV Haar级联分类器进行人脸检测
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size
                )
            else:
                logger.error("人脸检测器未初始化")
                return 0图片分类到不同文件夹
"""

import os
import shutil
import cv2
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_count_classifier_opencv.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceCountClassifierOpenCV:
    """使用OpenCV进行人脸检测和分类的类"""
    
    def __init__(self, cascade_path: Optional[str] = None, scale_factor: float = 1.1, 
                 min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)):
        """
        初始化分类器
        
        Args:
            cascade_path: Haar级联分类器文件路径
            scale_factor: 缩放因子
            min_neighbors: 最小邻居数
            min_size: 最小人脸尺寸
        """
        if cascade_path is None:
            cascade_path = '/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml'
        
        self.cascade_path = cascade_path
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.face_cascade = None
        self.load_cascade()
        
        # 支持的图片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def load_cascade(self):
        """加载Haar级联分类器"""
        try:
            logger.info(f"正在加载级联分类器: {self.cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            if self.face_cascade.empty():
                logger.error("级联分类器加载失败")
                raise Exception("无法加载级联分类器")
            
            logger.info("级联分类器加载成功")
        except Exception as e:
            logger.error(f"级联分类器加载失败: {e}")
            raise
    
    def detect_faces(self, image_path: str) -> int:
        """
        检测图片中的人脸数量
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            检测到的人脸数量
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return 0
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            face_count = len(faces)
            logger.info(f"图片 {os.path.basename(image_path)} 检测到 {face_count} 个人脸")
            
            return face_count
            
        except Exception as e:
            logger.error(f"检测图片 {image_path} 时出错: {e}")
            return 0
    
    def detect_faces_with_visualization(self, image_path: str, output_path: Optional[str] = None) -> int:
        """
        检测人脸并保存标注结果
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            
        Returns:
            检测到的人脸数量
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return 0
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # 在图片上标注人脸
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 添加人脸计数文本
            face_count = len(faces)
            cv2.putText(image, f'Faces: {face_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 保存结果
            if output_path:
                cv2.imwrite(output_path, image)
                logger.info(f"标注结果保存到: {output_path}")
            
            return face_count
            
        except Exception as e:
            logger.error(f"检测图片 {image_path} 时出错: {e}")
            return 0
    
    def create_output_directories(self, base_output_dir: str, max_faces: int = 10):
        """
        创建输出目录结构
        
        Args:
            base_output_dir: 基础输出目录
            max_faces: 最大人脸数
        """
        base_path = Path(base_output_dir)
        base_path.mkdir(exist_ok=True)
        
        # 创建不同人脸数的文件夹
        for i in range(max_faces + 1):
            face_dir = base_path / f"{i}_faces"
            face_dir.mkdir(exist_ok=True)
            logger.info(f"创建目录: {face_dir}")
        
        # 创建多人脸文件夹（超过max_faces的情况）
        many_faces_dir = base_path / "many_faces"
        many_faces_dir.mkdir(exist_ok=True)
        logger.info(f"创建目录: {many_faces_dir}")
    
    def classify_images(self, input_dir: str, output_dir: str, max_faces: int = 10, 
                       copy_mode: bool = False, visualize: bool = False):
        """
        分类图片到不同文件夹
        
        Args:
            input_dir: 输入图片目录
            output_dir: 输出目录
            max_faces: 最大人脸数分类
            copy_mode: True为复制模式，False为移动模式
            visualize: 是否保存可视化结果
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        self.create_output_directories(output_dir, max_faces)
        
        # 如果需要可视化，创建可视化目录
        viz_dir = None
        if visualize:
            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 统计信息
        classification_stats = {}
        
        # 处理每个图片
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"处理图片 {i}/{len(image_files)}: {image_file.name}")
            
            # 检测人脸数量
            if visualize and viz_dir:
                viz_path = viz_dir / f"viz_{image_file.name}"
                face_count = self.detect_faces_with_visualization(str(image_file), str(viz_path))
            else:
                face_count = self.detect_faces(str(image_file))
            
            # 确定目标文件夹
            if face_count <= max_faces:
                target_dir = Path(output_dir) / f"{face_count}_faces"
            else:
                target_dir = Path(output_dir) / "many_faces"
            
            # 移动或复制文件
            try:
                target_file = target_dir / image_file.name
                
                # 如果目标文件已存在，添加序号
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                if copy_mode:
                    shutil.copy2(str(image_file), str(target_file))
                    logger.info(f"复制 {image_file.name} 到 {target_dir.name}")
                else:
                    shutil.move(str(image_file), str(target_file))
                    logger.info(f"移动 {image_file.name} 到 {target_dir.name}")
                
                # 更新统计
                dir_name = target_dir.name
                classification_stats[dir_name] = classification_stats.get(dir_name, 0) + 1
                
            except Exception as e:
                logger.error(f"处理文件 {image_file.name} 时出错: {e}")
        
        # 打印统计信息
        logger.info("\n分类统计:")
        for dir_name, count in sorted(classification_stats.items()):
            logger.info(f"{dir_name}: {count} 张图片")
        
        logger.info(f"总共处理了 {len(image_files)} 张图片")
    
    def classify_single_image(self, image_path: str, output_dir: str, max_faces: int = 10, 
                             copy_mode: bool = False, visualize: bool = False):
        """
        分类单张图片
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            max_faces: 最大人脸数分类
            copy_mode: True为复制模式，False为移动模式
            visualize: 是否保存可视化结果
        """
        image_file = Path(image_path)
        
        if not image_file.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return
        
        # 创建输出目录
        self.create_output_directories(output_dir, max_faces)
        
        # 如果需要可视化，创建可视化目录
        viz_dir = None
        if visualize:
            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(exist_ok=True)
        
        # 检测人脸数量
        if visualize and viz_dir:
            viz_path = viz_dir / f"viz_{image_file.name}"
            face_count = self.detect_faces_with_visualization(str(image_file), str(viz_path))
        else:
            face_count = self.detect_faces(str(image_file))
        
        # 确定目标文件夹
        if face_count <= max_faces:
            target_dir = Path(output_dir) / f"{face_count}_faces"
        else:
            target_dir = Path(output_dir) / "many_faces"
        
        # 移动或复制文件
        try:
            target_file = target_dir / image_file.name
            
            # 如果目标文件已存在，添加序号
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
            
            logger.info(f"图片 {image_file.name} (检测到 {face_count} 个人脸) {action}到 {target_dir.name}")
            
        except Exception as e:
            logger.error(f"处理文件 {image_file.name} 时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用OpenCV按人脸数量分类图片')
    parser.add_argument('--cascade', '-c', 
                       default='/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml',
                       help='Haar级联分类器文件路径')
    parser.add_argument('--input', '-i', 
                       default='/home/zhiqics/sanjian/predata/test_images',
                       help='输入图片目录或单个图片文件')
    parser.add_argument('--output', '-o', 
                       default='/home/zhiqics/sanjian/predata/classified_images_opencv',
                       help='输出目录')
    parser.add_argument('--max-faces', type=int, default=10,
                       help='最大人脸数分类 (默认: 10)')
    parser.add_argument('--scale-factor', type=float, default=1.1,
                       help='检测缩放因子 (默认: 1.1)')
    parser.add_argument('--min-neighbors', type=int, default=5,
                       help='最小邻居数 (默认: 5)')
    parser.add_argument('--min-size', type=int, nargs=2, default=[30, 30],
                       help='最小人脸尺寸 [宽 高] (默认: 30 30)')
    parser.add_argument('--copy', action='store_true',
                       help='使用复制模式而不是移动模式')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='保存可视化结果')
    
    args = parser.parse_args()
    
    try:
        # 初始化分类器
        classifier = FaceCountClassifierOpenCV(
            cascade_path=args.cascade,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size)
        )
        
        # 检查输入是文件还是目录
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 处理单个文件
            logger.info(f"处理单个图片: {args.input}")
            classifier.classify_single_image(
                args.input, args.output, args.max_faces, 
                args.copy, args.visualize
            )
        elif input_path.is_dir():
            # 处理目录
            logger.info(f"处理目录: {args.input}")
            classifier.classify_images(
                args.input, args.output, args.max_faces, 
                args.copy, args.visualize
            )
        else:
            logger.error(f"输入路径不存在: {args.input}")
            return
        
        logger.info("分类完成!")
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
