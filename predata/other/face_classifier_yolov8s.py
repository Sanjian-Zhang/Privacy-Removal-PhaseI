#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 YOLOv8s 模型进行人脸检测和数量分类
专门针对 /home/zhiqics/sanjian/predata/models/yolov8s.pt 模型优化
"""

import os
import cv2
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import math

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查依赖库
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ ultralytics 库可用")
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("❌ ultralytics 库未安装，请运行: pip install ultralytics")
    exit(1)


class YOLOv8sFaceClassifier:
    """使用 YOLOv8s 模型进行人脸检测和分类，包含人脸质量评估"""
    
    def __init__(self, 
                 model_path: str = "/home/zhiqics/sanjian/predata/models/yolov8s.pt",
                 min_face_size: int = 30,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 use_gpu: bool = True,
                 face_quality_threshold: float = 0.6,  # 人脸质量阈值
                 filter_side_faces: bool = True,       # 是否过滤侧脸
                 filter_blurry_faces: bool = True):    # 是否过滤模糊人脸
        """
        初始化 YOLOv8s 人脸分类器
        
        Args:
            model_path: YOLOv8s 模型路径
            min_face_size: 最小人脸尺寸 (像素)
            confidence_threshold: 置信度阈值
            iou_threshold: IoU 阈值，用于非极大值抑制
            use_gpu: 是否使用 GPU
            face_quality_threshold: 人脸质量阈值 (0-1)
            filter_side_faces: 是否过滤侧脸和后脑勺
            filter_blurry_faces: 是否过滤模糊人脸
        """
        self.model_path = model_path
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_gpu = use_gpu
        self.face_quality_threshold = face_quality_threshold
        self.filter_side_faces = filter_side_faces
        self.filter_blurry_faces = filter_blurry_faces
        
        # 设备检查和配置
        self.device = self._setup_device()
        
        # 初始化模型
        self.model = self._load_model()
        
        # 初始化人脸质量评估器
        self._init_face_quality_detectors()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'no_faces': 0,
            'single_face': 0,
            'few_faces': 0,  # 2-3 张人脸
            'many_faces': 0,  # 4+ 张人脸
            'errors': 0,
            'filtered_low_quality': 0,  # 被质量过滤器过滤的人脸数量
            'filtered_side_faces': 0,   # 被侧脸过滤器过滤的数量
            'filtered_blurry': 0        # 被模糊过滤器过滤的数量
        }
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.use_gpu and torch.cuda.is_available():
            try:
                # 测试 CUDA
                test_tensor = torch.tensor([1.0]).cuda()
                device = 'cuda'
                logger.info(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
                del test_tensor
            except Exception as e:
                logger.warning(f"⚠️  GPU 不可用，切换到 CPU: {e}")
                device = 'cpu'
        else:
            device = 'cpu'
            logger.info("📱 使用 CPU 进行推理")
        
        return device
    
    def _init_face_quality_detectors(self):
        """初始化人脸质量评估器"""
        try:
            # 初始化 OpenCV 的人脸检测器（用于质量评估）
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'  # type: ignore
            )
            
            logger.info("✅ 人脸质量评估器初始化成功")
            
        except Exception as e:
            logger.warning(f"⚠️  人脸质量评估器初始化失败: {e}")
            self.filter_side_faces = False
    
    def _calculate_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        计算人脸质量分数
        
        Args:
            image: 原始图像
            bbox: 人脸边界框 (x1, y1, x2, y2)
            
        Returns:
            dict: 包含各种质量指标的字典
        """
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return {'overall_quality': 0.0, 'sharpness': 0.0, 'brightness': 0.0, 'contrast': 0.0}
        
        # 转换为灰度图
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        # 1. 清晰度评估（拉普拉斯算子）
        laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # 归一化到0-1
        
        # 2. 亮度评估
        mean_brightness = np.mean(face_gray)
        # 理想亮度范围 80-180
        if 80 <= mean_brightness <= 180:
            brightness_score = 1.0
        else:
            brightness_score = max(0.0, 1.0 - abs(mean_brightness - 130) / 130.0)
        
        # 3. 对比度评估
        contrast = np.std(face_gray)
        contrast_score = min(contrast / 60.0, 1.0)  # 归一化到0-1
        
        # 4. 综合质量分数
        overall_quality = (sharpness_score * 0.5 + brightness_score * 0.2 + contrast_score * 0.3)
        
        return {
            'overall_quality': float(overall_quality),
            'sharpness': float(sharpness_score),
            'brightness': float(brightness_score),
            'contrast': float(contrast_score)
        }
    
    def _is_frontal_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        判断是否为正面人脸，过滤侧脸和后脑勺
        
        Args:
            image: 原始图像
            bbox: 人脸边界框 (x1, y1, x2, y2)
            
        Returns:
            tuple: (是否为正面人脸, 正面程度分数)
        """
        if not self.filter_side_faces:
            return True, 1.0
        
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return False, 0.0
        
        # 转换为灰度图
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        try:
            # 使用正面人脸检测器
            frontal_faces = self.face_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 使用侧面人脸检测器
            profile_faces = self.profile_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            frontal_count = len(frontal_faces)
            profile_count = len(profile_faces)
            
            # 如果检测到正面人脸，计算正面程度
            if frontal_count > 0:
                # 正面人脸数量越多，正面程度越高
                frontal_score = min(1.0, frontal_count / (frontal_count + profile_count + 1))
                
                # 如果有侧脸检测，降低正面程度
                if profile_count > 0:
                    frontal_score *= 0.7
                
                # 正面程度大于0.3才认为是正面
                is_frontal = frontal_score > 0.3
                return is_frontal, frontal_score
            
            # 如果没有检测到任何人脸特征，可能是后脑勺或其他
            return False, 0.0
            
        except Exception:
            # 如果检测失败，保守地认为是正面
            return True, 0.5
    
    def _assess_face_pose(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        评估人脸姿态，检测是否为后脑勺或极度侧面
        
        Args:
            image: 原始图像
            bbox: 人脸边界框 (x1, y1, x2, y2)
            
        Returns:
            dict: 包含姿态评估结果的字典
        """
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return {'pose_quality': 0.0, 'is_back_head': True}
        
        # 转换为灰度图
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        # 简单的后脑勺检测：基于纹理和边缘特征
        # 1. 边缘检测
        edges = cv2.Canny(face_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. 纹理分析（标准差）
        texture_std = np.std(face_gray)
        
        # 3. 对称性检查
        h, w = face_gray.shape
        left_half = face_gray[:, :w//2]
        right_half = cv2.flip(face_gray[:, w//2:], 1)
        
        # 调整尺寸以匹配
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # 计算对称性
        if left_half.shape == right_half.shape:
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry_score = 0.5
        
        # 综合判断
        # 正常人脸：边缘密度适中，纹理丰富，有一定对称性
        # 后脑勺：边缘密度低，纹理单一，对称性差
        
        pose_score = (edge_density * 2 + texture_std / 100.0 + symmetry_score) / 3.0
        pose_score = min(1.0, pose_score)
        
        # 阈值判断是否为后脑勺
        is_back_head = pose_score < 0.3
        
        return {
            'pose_quality': float(pose_score),
            'is_back_head': bool(is_back_head),
            'edge_density': float(edge_density),
            'texture_std': float(texture_std),
            'symmetry_score': float(symmetry_score)
        }
    
    def _load_model(self) -> YOLO:
        """加载 YOLOv8s 模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ 模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载模型
            model = YOLO(self.model_path)
            
            # 设置设备
            model.to(self.device)
            
            logger.info(f"✅ YOLOv8s 模型加载成功")
            logger.info(f"   📁 模型路径: {self.model_path}")
            logger.info(f"   🖥️  运行设备: {self.device}")
            logger.info(f"   🎯 置信度阈值: {self.confidence_threshold}")
            logger.info(f"   📏 最小人脸尺寸: {self.min_face_size}px")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e
    
    def detect_faces(self, image_path: str) -> Tuple[int, List[Dict]]:
        """
        检测图片中的人脸，包含质量评估和过滤
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (高质量人脸数量, 检测结果列表)
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return 0, []
            
            # 读取图片获取尺寸信息
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return 0, []
            
            height, width = img.shape[:2]
            
            # 使用 YOLOv8s 进行推理
            results = self.model(
                image_path,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False  # 减少输出信息
            )
            
            if not results:
                return 0, []
            
            all_detections = []
            
            # 处理检测结果
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # 计算人脸尺寸
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        # 基本过滤：尺寸和边缘检查
                        if (face_width >= self.min_face_size and 
                            face_height >= self.min_face_size and
                            x1 > width * 0.01 and y1 > height * 0.01 and  # 不在图片边缘
                            x2 < width * 0.99 and y2 < height * 0.99):
                            
                            bbox = (int(x1), int(y1), int(x2), int(y2))
                            
                            # 计算人脸质量
                            quality_metrics = self._calculate_face_quality(img, bbox)
                            
                            # 检查是否为正面人脸
                            is_frontal, frontal_score = self._is_frontal_face(img, bbox)
                            
                            # 评估人脸姿态
                            pose_metrics = self._assess_face_pose(img, bbox)
                            
                            # 综合质量评分
                            overall_score = (
                                quality_metrics['overall_quality'] * 0.4 +
                                frontal_score * 0.3 +
                                pose_metrics['pose_quality'] * 0.3
                            )
                            
                            detection_info = {
                                'bbox': bbox,
                                'confidence': float(confidence),
                                'width': int(face_width),
                                'height': int(face_height),
                                'area': int(face_width * face_height),
                                'overall_score': overall_score,
                                'quality_metrics': quality_metrics,
                                'frontal_score': frontal_score,
                                'is_frontal': is_frontal,
                                'pose_metrics': pose_metrics,
                                'is_high_quality': False  # 将在后面设置
                            }
                            
                            all_detections.append(detection_info)
            
            # 按综合质量分数排序
            all_detections.sort(key=lambda x: x['overall_score'], reverse=True)
            
            # 应用质量过滤
            high_quality_faces = []
            filtered_stats = {'low_quality': 0, 'side_faces': 0, 'blurry': 0, 'back_head': 0}
            
            for detection in all_detections:
                should_keep = True
                filter_reasons = []
                
                # 1. 综合质量检查
                if detection['overall_score'] < self.face_quality_threshold:
                    should_keep = False
                    filter_reasons.append('low_overall_quality')
                    filtered_stats['low_quality'] += 1
                
                # 2. 侧脸过滤
                if self.filter_side_faces and not detection['is_frontal']:
                    should_keep = False
                    filter_reasons.append('side_face')
                    filtered_stats['side_faces'] += 1
                
                # 3. 后脑勺过滤
                if detection['pose_metrics']['is_back_head']:
                    should_keep = False
                    filter_reasons.append('back_head')
                    filtered_stats['back_head'] += 1
                
                # 4. 模糊度过滤
                if (self.filter_blurry_faces and 
                    detection['quality_metrics']['sharpness'] < 0.3):
                    should_keep = False
                    filter_reasons.append('blurry')
                    filtered_stats['blurry'] += 1
                
                if should_keep:
                    detection['is_high_quality'] = True
                    high_quality_faces.append(detection)
                else:
                    detection['filter_reasons'] = filter_reasons
            
            # 更新统计信息
            self.stats['filtered_low_quality'] += filtered_stats['low_quality']
            self.stats['filtered_side_faces'] += filtered_stats['side_faces']
            self.stats['filtered_blurry'] += filtered_stats['blurry']
            
            logger.debug(f"检测到 {len(all_detections)} 张人脸，过滤后保留 {len(high_quality_faces)} 张高质量人脸")
            
            return len(high_quality_faces), high_quality_faces
            
        except Exception as e:
            logger.error(f"检测出错 {os.path.basename(image_path)}: {str(e)}")
            return 0, []
    
    def classify_by_face_count(self, face_count: int) -> str:
        """
        根据人脸数量进行分类
        
        Args:
            face_count: 检测到的人脸数量
            
        Returns:
            str: 分类类别
        """
        if face_count == 0:
            return "no_faces"
        elif face_count == 1:
            return "single_face"
        elif face_count <= 3:
            return "few_faces"  # 2-3 张人脸
        else:
            return "many_faces"  # 4+ 张人脸
    
    def test_single_image(self, image_path: str) -> None:
        """测试单张图片的检测效果，显示详细的质量评估信息"""
        logger.info(f"🧪 测试图片: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            logger.error(f"❌ 文件不存在: {image_path}")
            return
        
        # 检测人脸
        face_count, detections = self.detect_faces(image_path)
        category = self.classify_by_face_count(face_count)
        
        logger.info(f"📊 检测结果:")
        logger.info(f"   👥 高质量人脸数量: {face_count}")
        logger.info(f"   🏷️  分类类别: {category}")
        
        if face_count > 0:
            logger.info(f"   📝 详细信息:")
            for i, face in enumerate(detections, 1):
                bbox = face['bbox']
                conf = face['confidence']
                size = f"{face['width']}×{face['height']}"
                area = face['area']
                overall_score = face.get('overall_score', 0)
                
                logger.info(f"      人脸 {i}:")
                logger.info(f"         位置: {bbox}")
                logger.info(f"         YOLO置信度: {conf:.3f}")
                logger.info(f"         综合质量分数: {overall_score:.3f}")
                logger.info(f"         尺寸: {size} (面积: {area})")
                
                if 'is_frontal' in face:
                    logger.info(f"         是否正面: {'是' if face['is_frontal'] else '否'} (分数: {face.get('frontal_score', 0):.3f})")
                
                # 质量指标
                if 'quality_metrics' in face:
                    quality = face['quality_metrics']
                    logger.info(f"         清晰度: {quality['sharpness']:.3f}")
                    logger.info(f"         亮度: {quality['brightness']:.3f}")
                    logger.info(f"         对比度: {quality['contrast']:.3f}")
                
                # 姿态指标
                if 'pose_metrics' in face:
                    pose = face['pose_metrics']
                    logger.info(f"         姿态质量: {pose['pose_quality']:.3f}")
                    logger.info(f"         是否后脑勺: {'是' if pose['is_back_head'] else '否'}")
                
                if not face.get('is_high_quality', True) and 'filter_reasons' in face:
                    reasons = ', '.join(face['filter_reasons'])
                    logger.info(f"         ⚠️  过滤原因: {reasons}")
                
                logger.info("")
        else:
            logger.info("   ❌ 未检测到高质量人脸")
        
        # 显示过滤统计
        logger.info(f"   📈 本次过滤统计:")
        logger.info(f"      质量过滤设置: 阈值={self.face_quality_threshold}, 过滤侧脸={self.filter_side_faces}, 过滤模糊={self.filter_blurry_faces}")
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str,
                         copy_files: bool = True,
                         create_report: bool = True) -> Dict:
        """
        处理整个目录的图片
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            copy_files: 是否复制文件到分类目录
            create_report: 是否创建详细报告
            
        Returns:
            dict: 处理结果统计
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {input_dir}")
            return {}
        
        # 创建输出目录结构
        categories = {
            "no_faces": output_path / "0_faces" / "no_faces",
            "single_face": output_path / "1_face" / "single_face", 
            "few_faces": output_path / "2-3_faces" / "few_faces",
            "many_faces": output_path / "4+_faces" / "many_faces"
        }
        
        if copy_files:
            for category_path in categories.values():
                category_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 创建目录: {category_path}")
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.JPG', '.JPEG', '.PNG'}
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
        
        if not image_files:
            logger.error(f"❌ 在 {input_dir} 中未找到图片文件")
            return {}
        
        logger.info(f"📷 找到 {len(image_files)} 张图片")
        logger.info(f"🚀 开始处理...")
        
        # 详细处理记录
        processing_log = []
        
        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            try:
                logger.info(f"[{i:4d}/{len(image_files)}] 处理: {image_file.name}")
                
                # 检测人脸
                face_count, detections = self.detect_faces(str(image_file))
                category = self.classify_by_face_count(face_count)
                
                # 更新统计
                self.stats['total_processed'] += 1
                if category == "no_faces":
                    self.stats['no_faces'] += 1
                    logger.info(f"   ❌ 无人脸")
                elif category == "single_face":
                    self.stats['single_face'] += 1
                    logger.info(f"   👤 单人脸")
                elif category == "few_faces":
                    self.stats['few_faces'] += 1
                    logger.info(f"   👥 少量人脸 ({face_count} 张)")
                else:  # many_faces
                    self.stats['many_faces'] += 1
                    logger.info(f"   👨‍👩‍👧‍👦 多人脸 ({face_count} 张)")
                
                # 记录处理结果
                processing_log.append({
                    'filename': image_file.name,
                    'face_count': face_count,
                    'category': category,
                    'detections': detections
                })
                
                # 复制文件到相应目录
                if copy_files and category in categories:
                    dest_dir = categories[category]
                    dest_file = dest_dir / image_file.name
                    
                    # 处理文件名冲突
                    counter = 1
                    original_dest = dest_file
                    while dest_file.exists():
                        name_stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_file = dest_dir / f"{name_stem}_{counter}{suffix}"
                        counter += 1
                    
                    # 复制文件（保持原始质量）
                    shutil.copy2(str(image_file), str(dest_file))
                
            except Exception as e:
                logger.error(f"   ❌ 处理出错: {str(e)}")
                self.stats['errors'] += 1
                processing_log.append({
                    'filename': image_file.name,
                    'face_count': 0,
                    'category': 'error',
                    'error': str(e)
                })
        
        # 创建详细报告
        if create_report:
            self._create_report(output_path, processing_log)
        
        # 打印最终统计
        self._print_final_stats()
        
        return self.stats.copy()
    
    def _create_report(self, output_path: Path, processing_log: List[Dict]) -> None:
        """创建处理报告"""
        report_path = output_path / f"face_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("YOLOv8s 人脸检测分类报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"使用设备: {self.device}\n")
                f.write(f"置信度阈值: {self.confidence_threshold}\n")
                f.write(f"最小人脸尺寸: {self.min_face_size}px\n")
                f.write("\n统计结果:\n")
                f.write(f"总图片数: {self.stats['total_processed']}\n")
                f.write(f"无人脸: {self.stats['no_faces']}\n")
                f.write(f"单人脸: {self.stats['single_face']}\n")
                f.write(f"少量人脸(2-3张): {self.stats['few_faces']}\n")
                f.write(f"多人脸(4+张): {self.stats['many_faces']}\n")
                f.write(f"处理错误: {self.stats['errors']}\n")
                f.write("\n详细处理记录:\n")
                f.write("-" * 50 + "\n")
                
                for record in processing_log:
                    f.write(f"文件名: {record['filename']}\n")
                    f.write(f"人脸数量: {record['face_count']}\n")
                    f.write(f"分类: {record['category']}\n")
                    
                    if 'error' in record:
                        f.write(f"错误: {record['error']}\n")
                    elif record['detections']:
                        f.write("检测详情:\n")
                        for i, det in enumerate(record['detections'], 1):
                            f.write(f"  人脸{i}: 位置{det['bbox']}, 置信度{det['confidence']:.3f}, 尺寸{det['width']}×{det['height']}\n")
                    
                    f.write("-" * 30 + "\n")
            
            logger.info(f"📄 详细报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"创建报告失败: {e}")
    
    def _print_final_stats(self) -> None:
        """打印最终统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 处理完成！")
        logger.info("=" * 60)
        logger.info(f"📊 总图片数: {self.stats['total_processed']}")
        logger.info(f"❌ 无人脸图片: {self.stats['no_faces']} ({self.stats['no_faces']/max(self.stats['total_processed'],1)*100:.1f}%)")
        logger.info(f"👤 单人脸图片: {self.stats['single_face']} ({self.stats['single_face']/max(self.stats['total_processed'],1)*100:.1f}%)")
        logger.info(f"👥 少量人脸图片(2-3张): {self.stats['few_faces']} ({self.stats['few_faces']/max(self.stats['total_processed'],1)*100:.1f}%)")
        logger.info(f"👨‍👩‍👧‍👦 多人脸图片(4+张): {self.stats['many_faces']} ({self.stats['many_faces']/max(self.stats['total_processed'],1)*100:.1f}%)")
        logger.info(f"⚠️  处理错误: {self.stats['errors']}")
        logger.info("=" * 40)
        logger.info("🔍 质量过滤统计:")
        logger.info(f"   🚫 低质量过滤: {self.stats['filtered_low_quality']}")
        logger.info(f"   🚫 侧脸过滤: {self.stats['filtered_side_faces']}")
        logger.info(f"   🚫 模糊过滤: {self.stats['filtered_blurry']}")
        total_filtered = (self.stats['filtered_low_quality'] + 
                         self.stats['filtered_side_faces'] + 
                         self.stats['filtered_blurry'])
        logger.info(f"   🚫 总过滤数: {total_filtered}")
        logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("🚀 YOLOv8s 人脸检测分类器 (支持质量过滤)")
    logger.info("=" * 50)
    
    # 配置参数
    MODEL_PATH = "/home/zhiqics/sanjian/predata/models/yolov8s.pt"
    INPUT_DIR = "/home/zhiqics/sanjian/predata/output_frames22"  # 修改为您的输入目录
    OUTPUT_DIR = "/home/zhiqics/sanjian/predata/face_classification_results"
    
    # 检测参数
    MIN_FACE_SIZE = 40              # 最小人脸尺寸 (增加以过滤小脸)
    CONFIDENCE_THRESHOLD = 0.3      # YOLO置信度阈值
    IOU_THRESHOLD = 0.45           # IoU 阈值
    USE_GPU = True                 # 是否使用 GPU
    
    # 质量过滤参数
    FACE_QUALITY_THRESHOLD = 0.5   # 人脸综合质量阈值 (0-1)
    FILTER_SIDE_FACES = True       # 过滤侧脸和后脑勺
    FILTER_BLURRY_FACES = True     # 过滤模糊人脸
    
    logger.info("🔧 配置参数:")
    logger.info(f"   📁 模型路径: {MODEL_PATH}")
    logger.info(f"   📁 输入目录: {INPUT_DIR}")
    logger.info(f"   📁 输出目录: {OUTPUT_DIR}")
    logger.info(f"   📏 最小人脸尺寸: {MIN_FACE_SIZE}px")
    logger.info(f"   🎯 YOLO置信度阈值: {CONFIDENCE_THRESHOLD}")
    logger.info(f"   🎯 质量过滤阈值: {FACE_QUALITY_THRESHOLD}")
    logger.info(f"   🚫 过滤侧脸: {'是' if FILTER_SIDE_FACES else '否'}")
    logger.info(f"   🚫 过滤模糊: {'是' if FILTER_BLURRY_FACES else '否'}")
    
    # 创建分类器
    try:
        classifier = YOLOv8sFaceClassifier(
            model_path=MODEL_PATH,
            min_face_size=MIN_FACE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            use_gpu=USE_GPU,
            face_quality_threshold=FACE_QUALITY_THRESHOLD,
            filter_side_faces=FILTER_SIDE_FACES,
            filter_blurry_faces=FILTER_BLURRY_FACES
        )
    except Exception as e:
        logger.error(f"创建分类器失败: {e}")
        return
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ 输入目录不存在: {INPUT_DIR}")
        logger.info("请确保输入目录存在并包含图片文件")
        return
    
    # 检查是否有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(INPUT_DIR).rglob(f'*{ext}'))
    
    if not image_files:
        logger.error(f"❌ 在 {INPUT_DIR} 中未找到图片文件")
        return
    
    logger.info(f"📷 找到 {len(image_files)} 张图片")
    
    # 测试第一张图片
    if image_files:
        logger.info("\n" + "=" * 40)
        logger.info("🧪 测试第一张图片")
        logger.info("=" * 40)
        classifier.test_single_image(str(image_files[0]))
        
        # 询问是否继续
        print("\n" + "=" * 40)
        response = input("❓ 是否继续处理所有图片？(y/n): ").lower().strip()
        if response not in ['y', 'yes', '是', '继续']:
            logger.info("👋 已取消处理")
            return
    
    # 开始批量处理
    logger.info(f"\n📁 输入目录: {INPUT_DIR}")
    logger.info(f"📁 输出目录: {OUTPUT_DIR}")
    
    results = classifier.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        copy_files=True,      # 复制文件到分类目录
        create_report=True    # 创建详细报告
    )
    
    logger.info(f"✅ 处理完成，结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
