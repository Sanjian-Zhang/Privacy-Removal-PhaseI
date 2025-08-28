#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸数量分类器 - 使用YOLOv8模型
按照检测到的人脸数量将图片分类到不同文件夹
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Tuple, Dict, Optional
import logging

# RetinaFace导入
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("警告: RetinaFace未安装，将使用OpenCV人脸检测")
    print("请运行: pip install retina-face")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_count_classifier_yolo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceCountClassifier:
    """使用YOLOv8和RetinaFace进行正脸检测和分类的类"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 cascade_path: Optional[str] = None, use_frontal_face: bool = True,
                 use_retinaface: bool = True):
        """
        初始化分类器
        
        Args:
            model_path: YOLOv8模型文件路径
            confidence_threshold: 置信度阈值
            cascade_path: Haar级联分类器文件路径（备用）
            use_frontal_face: 是否使用正脸检测
            use_retinaface: 是否优先使用RetinaFace
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_frontal_face = use_frontal_face
        self.use_retinaface = use_retinaface and RETINAFACE_AVAILABLE
        self.model = None
        self.face_cascade = None
        
        # 默认的级联分类器路径（作为备选方案）
        if cascade_path is None:
            cascade_path = '/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml'
        self.cascade_path = cascade_path
        
        self.load_model()
        
        # 如果RetinaFace不可用，回退到OpenCV
        if self.use_frontal_face and not self.use_retinaface:
            self.load_face_cascade()
        
        # 支持的图片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def load_face_cascade(self):
        """加载Haar级联分类器"""
        try:
            logger.info(f"正在加载人脸检测器: {self.cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("级联分类器加载失败，将禁用正脸检测")
                self.use_frontal_face = False
            else:
                logger.info("人脸检测器加载成功")
        except Exception as e:
            logger.error(f"人脸检测器加载失败: {e}")
            self.use_frontal_face = False
    
    def detect_frontal_faces_retinaface(self, image_path: str) -> int:
        """
        使用RetinaFace检测正脸数量
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            检测到的正脸数量
        """
        try:
            if not self.use_retinaface or not RETINAFACE_AVAILABLE:
                return 0
            
            # 使用RetinaFace检测人脸
            if RETINAFACE_AVAILABLE:
                from retinaface import RetinaFace
                faces = RetinaFace.detect_faces(image_path)
            else:
                return 0
            
            if faces is None or len(faces) == 0:
                return 0
            
            # 计算正脸数量（基于人脸角度判断）
            frontal_count = 0
            for face_key, face_data in faces.items():
                # 获取人脸区域和关键点
                facial_area = face_data.get('facial_area', [])
                landmarks = face_data.get('landmarks', {})
                
                if landmarks:
                    # 计算人脸角度来判断是否为正脸
                    if self._is_frontal_face(landmarks):
                        frontal_count += 1
                else:
                    # 如果没有关键点信息，直接计数
                    frontal_count += 1
            
            logger.info(f"RetinaFace检测到 {len(faces)} 个人脸，其中 {frontal_count} 个正脸")
            return frontal_count
            
        except Exception as e:
            logger.error(f"RetinaFace检测失败 {image_path}: {e}")
            return 0
    
    def _is_frontal_face(self, landmarks: dict) -> bool:
        """
        根据关键点判断是否为正脸
        
        Args:
            landmarks: 人脸关键点字典
            
        Returns:
            是否为正脸
        """
        try:
            # 获取关键点
            left_eye = landmarks.get('left_eye', None)
            right_eye = landmarks.get('right_eye', None)
            nose = landmarks.get('nose', None)
            
            if left_eye is None or right_eye is None or nose is None:
                return True  # 如果关键点不完整，默认认为是正脸
            
            # 计算眼睛中心点
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            
            # 计算鼻子到眼睛中心的偏移
            nose_offset_x = abs(nose[0] - eye_center_x)
            eye_distance = abs(left_eye[0] - right_eye[0])
            
            # 如果鼻子在眼睛中心附近，认为是正脸
            # 阈值可以调整，0.3表示鼻子偏移不超过眼距的30%
            frontal_threshold = 0.3
            if eye_distance > 0:
                offset_ratio = nose_offset_x / eye_distance
                return offset_ratio < frontal_threshold
            
            return True
            
        except Exception as e:
            logger.error(f"正脸判断失败: {e}")
            return True  # 出错时默认认为是正脸
    
    def detect_frontal_faces(self, image_path: str) -> int:
        """
        使用OpenCV检测正脸数量（备用方法）
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            检测到的正脸数量
        """
        try:
            if self.face_cascade is None:
                return 0
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return 0
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测正脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return len(faces)
            
        except Exception as e:
            logger.error(f"OpenCV正脸检测失败 {image_path}: {e}")
            return 0
    
    def detect_persons_yolo(self, image_path: str) -> int:
        """
        使用YOLO检测人体数量
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            检测到的人体数量
        """
        try:
            if self.model is None:
                logger.error("YOLO模型未正确加载")
                return 0
                
            # 使用YOLOv8进行检测
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # 计算人体数量（COCO数据集中，class 0 是person）
            person_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # COCO数据集中，class 0 是person
                        if class_id == 0 and confidence >= self.confidence_threshold:
                            person_count += 1
            
            return person_count
            
        except Exception as e:
            logger.error(f"YOLO检测失败 {image_path}: {e}")
            return 0
    
    def detect_faces(self, image_path: str) -> Dict[str, int]:
        """
        综合检测图片中的人脸数量
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            包含不同检测方法结果的字典
        """
        results = {
            'frontal_faces': 0,
            'persons_yolo': 0,
            'total_count': 0
        }
        
        try:
            # 正脸检测
            detection_method = ""
            if self.use_frontal_face:
                if self.use_retinaface:
                    # 优先使用RetinaFace
                    frontal_count = self.detect_frontal_faces_retinaface(image_path)
                    detection_method = "RetinaFace"
                else:
                    # 使用OpenCV作为备选
                    frontal_count = self.detect_frontal_faces(image_path)
                    detection_method = "OpenCV"
                
                results['frontal_faces'] = frontal_count
            
            # YOLO人体检测
            person_count = self.detect_persons_yolo(image_path)
            results['persons_yolo'] = person_count
            
            # 计算总数：优先使用正脸检测结果，如果没有正脸则使用YOLO结果
            if self.use_frontal_face and results['frontal_faces'] > 0:
                results['total_count'] = results['frontal_faces']
                final_method = f"正脸检测({detection_method})"
            else:
                results['total_count'] = results['persons_yolo']
                final_method = "YOLO人体检测"
            
            logger.info(f"图片 {os.path.basename(image_path)} - "
                       f"正脸: {results['frontal_faces']}, "
                       f"YOLO人体: {results['persons_yolo']}, "
                       f"最终计数: {results['total_count']} ({final_method})")
            
            return results
            
        except Exception as e:
            logger.error(f"检测图片 {image_path} 时出错: {e}")
            return results
    
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
    
    def classify_images(self, input_dir: str, output_dir: str, max_faces: int = 10):
        """
        分类图片到不同文件夹
        
        Args:
            input_dir: 输入图片目录
            output_dir: 输出目录
            max_faces: 最大人脸数分类
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        self.create_output_directories(output_dir, max_faces)
        
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
            detection_results = self.detect_faces(str(image_file))
            face_count = detection_results['total_count']
            
            # 确定目标文件夹
            if face_count <= max_faces:
                target_dir = Path(output_dir) / f"{face_count}_faces"
            else:
                target_dir = Path(output_dir) / "many_faces"
            
            # 移动文件
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
                
                shutil.move(str(image_file), str(target_file))
                logger.info(f"移动 {image_file.name} 到 {target_dir.name}")
                
                # 更新统计
                dir_name = target_dir.name
                classification_stats[dir_name] = classification_stats.get(dir_name, 0) + 1
                
            except Exception as e:
                logger.error(f"移动文件 {image_file.name} 时出错: {e}")
        
        # 打印统计信息
        logger.info("\n分类统计:")
        for dir_name, count in sorted(classification_stats.items()):
            logger.info(f"{dir_name}: {count} 张图片")
    
    def classify_single_image(self, image_path: str, output_dir: str, max_faces: int = 10):
        """
        分类单张图片
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            max_faces: 最大人脸数分类
        """
        image_file = Path(image_path)
        
        if not image_file.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return
        
        # 创建输出目录
        self.create_output_directories(output_dir, max_faces)
        
        # 检测人脸数量
        detection_results = self.detect_faces(str(image_file))
        face_count = detection_results['total_count']
        
        # 确定目标文件夹
        if face_count <= max_faces:
            target_dir = Path(output_dir) / f"{face_count}_faces"
        else:
            target_dir = Path(output_dir) / "many_faces"
        
        # 移动文件
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
            
            shutil.move(str(image_file), str(target_file))
            logger.info(f"图片 {image_file.name} (检测到 {face_count} 个人脸) 移动到 {target_dir.name}")
            
        except Exception as e:
            logger.error(f"移动文件 {image_file.name} 时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用YOLOv8和RetinaFace按人脸数量分类图片')
    parser.add_argument('--model', '-m', 
                       default='/home/zhiqics/sanjian/predata/models/yolov8s.pt',
                       help='YOLOv8模型文件路径')
    parser.add_argument('--input', '-i', 
                       default='/home/zhiqics/sanjian/predata/output_frames70/0_faces',
                       help='输入图片目录或单个图片文件')
    parser.add_argument('--output', '-o', 
                       default='/home/zhiqics/sanjian/predata/classified_images70',
                       help='输出目录')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--max-faces', type=int, default=10,
                       help='最大人脸数分类 (默认: 10)')
    parser.add_argument('--no-frontal', action='store_true',
                       help='禁用正脸检测，只使用YOLO')
    parser.add_argument('--no-retinaface', action='store_true',
                       help='禁用RetinaFace，使用OpenCV进行人脸检测')
    
    args = parser.parse_args()
    
    try:
        # 检查RetinaFace可用性
        if not RETINAFACE_AVAILABLE and not args.no_frontal:
            logger.warning("RetinaFace不可用，需要安装: pip install retina-face")
            logger.info("将使用OpenCV进行人脸检测")
        
        # 初始化分类器
        classifier = FaceCountClassifier(
            model_path=args.model, 
            confidence_threshold=args.confidence,
            use_frontal_face=not args.no_frontal,
            use_retinaface=not args.no_retinaface
        )
        
        # 检查输入是文件还是目录
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 处理单个文件
            logger.info(f"处理单个图片: {args.input}")
            classifier.classify_single_image(args.input, args.output, args.max_faces)
        elif input_path.is_dir():
            # 处理目录
            logger.info(f"处理目录: {args.input}")
            classifier.classify_images(args.input, args.output, args.max_faces)
        else:
            logger.error(f"输入路径不存在: {args.input}")
            return
        
        logger.info("分类完成!")
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
