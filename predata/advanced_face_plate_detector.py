#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级人脸车牌检测器
功能要求：
1. 先用 YOLOv8s 识别图片中人脸数量
2. 如果正脸数量不满足4张要求，直接移动到相应文件夹
3. 用框的大小判断脸是近景还是远景，忽略远景
4. 用 RetinaFace 检测正脸
5. 只要图片中有4张近景正脸直接满足条件
6. 如果少于4张近景正脸，判断是否有清晰的车牌和文字
7. 考虑GPU0和1都可以使用
8. 处理图片时不压缩图片
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
from collections import defaultdict

# 设置环境变量 - 必须在导入深度学习库之前
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 启用GPU 0和1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # 减少TensorFlow日志输出

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_face_plate_detector.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查并导入依赖库"""
    missing_deps = []
    imported_modules = {}
    
    # 检查RetinaFace
    try:
        from retinaface import RetinaFace
        imported_modules['RetinaFace'] = RetinaFace
        logger.info("✅ RetinaFace 库导入成功")
    except ImportError as e:
        logger.error(f"❌ RetinaFace 库导入失败: {e}")
        missing_deps.append("retina-face")
    
    # 检查YOLO
    try:
        from ultralytics import YOLO
        imported_modules['YOLO'] = YOLO
        logger.info("✅ YOLO 库导入成功")
    except ImportError as e:
        logger.error(f"❌ YOLO 库导入失败: {e}")
        missing_deps.append("ultralytics")
    
    # 检查EasyOCR
    try:
        import easyocr
        imported_modules['easyocr'] = easyocr
        logger.info("✅ EasyOCR 库导入成功")
    except ImportError as e:
        logger.error(f"❌ EasyOCR 库导入失败: {e}")
        missing_deps.append("easyocr")
    
    # 检查torch
    try:
        import torch
        imported_modules['torch'] = torch
        logger.info("✅ PyTorch 库导入成功")
    except ImportError as e:
        logger.error(f"❌ PyTorch 库导入失败: {e}")
        missing_deps.append("torch")
    
    if missing_deps:
        logger.error(f"❌ 缺少依赖库: {', '.join(missing_deps)}")
        logger.error("请安装缺少的库:")
        for dep in missing_deps:
            logger.error(f"  pip install {dep}")
        return None
    
    return imported_modules

# 检查并导入依赖
modules = check_dependencies()
if modules is None:
    exit(1)

# 从检查结果中获取模块
RetinaFace = modules['RetinaFace']
YOLO = modules['YOLO']
easyocr = modules['easyocr']
torch = modules['torch']

class Config:
    """配置类"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames87'  # 输入目录
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames87'  # 输出基础目录
    
    # 模型路径
    YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
    LICENSE_PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 检测阈值
    REQUIRED_CLOSE_FRONTAL_FACES = 4      # 需要的近景正脸数量
    MIN_FACE_CONFIDENCE_YOLO = 0.5        # YOLO人脸最小置信度
    MIN_FACE_CONFIDENCE_RETINA = 0.8      # RetinaFace最小置信度
    MIN_PLATE_CONFIDENCE = 0.5            # 车牌最小置信度
    MIN_TEXT_CONFIDENCE = 0.5             # 文字最小置信度
    YAW_ANGLE_THRESHOLD = 30.0            # yaw角度阈值（正脸）
    
    # 近景判断参数
    MIN_FACE_SIZE = 80                    # 最小人脸尺寸（像素）
    CLOSE_UP_FACE_RATIO = 0.08           # 近景人脸最小面积比例
    MIN_FACE_AREA = 6400                 # 最小人脸面积（80x80）
    
    # 图像质量保护
    PRESERVE_IMAGE_QUALITY = True
    IMAGE_READ_FLAGS = cv2.IMREAD_COLOR
    JPEG_QUALITY = 100
    PNG_COMPRESSION = 0
    
    # GPU配置
    USE_GPU = True
    PREFERRED_GPU = 0                     # 优先使用GPU 0
    FALLBACK_GPU = 1                      # 备用GPU 1
    
    # 文件格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 性能参数
    GC_FREQUENCY = 50
    PROGRESS_UPDATE_FREQUENCY = 25
    MAX_PROCESSING_TIME_PER_IMAGE = 60
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'satisfied_4_faces': os.path.join(cls.OUTPUT_BASE_DIR, "satisfied_4_close_frontal_faces"),
            'satisfied_with_plate': os.path.join(cls.OUTPUT_BASE_DIR, "satisfied_with_plate_text"),
            'insufficient_faces': os.path.join(cls.OUTPUT_BASE_DIR, "insufficient_faces"),
            'no_faces': os.path.join(cls.OUTPUT_BASE_DIR, "no_faces"),
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis"),
        }

class AdvancedFacePlateDetector:
    """高级人脸车牌检测器"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化检测器"""
        self.config = config or Config()
        self.start_time = time.time()
        
        # 统计信息
        self.stats = {
            'satisfied_4_faces': 0,        # 4张近景正脸满足条件
            'satisfied_with_plate': 0,     # 通过车牌/文字满足条件
            'insufficient_faces': 0,       # 人脸不够
            'no_faces': 0,                 # 无人脸
            'failed': 0,                   # 处理失败
        }
        
        # 详细分析结果
        self.analysis_results = []
        
        # 获取输出目录
        self.output_dirs = self.config.get_output_dirs()
        
        # 设备配置
        self.device = self._setup_device()
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 初始化模型
        self._initialize_models()
        
        logger.info("🚀 高级人脸车牌检测器初始化完成")
        logger.info(f"📸 图像质量保护: 已启用，确保处理过程中不压缩图片")
        logger.info(f"🎯 需要近景正脸数量: {self.config.REQUIRED_CLOSE_FRONTAL_FACES}")
        logger.info(f"💻 使用设备: {self.device}")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        try:
            if self.config.USE_GPU and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"🔍 检测到 {device_count} 个GPU设备")
                
                # 尝试优先GPU
                if self.config.PREFERRED_GPU < device_count:
                    device = f'cuda:{self.config.PREFERRED_GPU}'
                    gpu_name = torch.cuda.get_device_name(self.config.PREFERRED_GPU)
                    gpu_memory = torch.cuda.get_device_properties(self.config.PREFERRED_GPU).total_memory / 1024**3
                    logger.info(f"🚀 使用优先GPU: {gpu_name} (设备 {self.config.PREFERRED_GPU})")
                    logger.info(f"🔥 GPU显存: {gpu_memory:.1f} GB")
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    return device
                
                # 尝试备用GPU
                elif self.config.FALLBACK_GPU < device_count:
                    device = f'cuda:{self.config.FALLBACK_GPU}'
                    gpu_name = torch.cuda.get_device_name(self.config.FALLBACK_GPU)
                    logger.info(f"🚀 使用备用GPU: {gpu_name} (设备 {self.config.FALLBACK_GPU})")
                    torch.cuda.empty_cache()
                    return device
                
                else:
                    logger.warning(f"⚠️  指定的GPU设备不可用，使用CPU模式")
                    return 'cpu'
            else:
                logger.info("💻 使用CPU模式")
                return 'cpu'
                
        except Exception as e:
            logger.error(f"❌ 设备设置失败: {e}")
            return 'cpu'
    
    def _create_output_dirs(self):
        """创建输出目录"""
        for name, dir_path in self.output_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 创建目录: {dir_path}")
            except Exception as e:
                logger.error(f"❌ 创建目录失败 {dir_path}: {e}")
                raise
    
    def _initialize_models(self):
        """初始化检测模型"""
        try:
            # 检查模型文件
            if not os.path.exists(self.config.YOLOV8S_MODEL_PATH):
                raise FileNotFoundError(f"YOLOv8s模型文件不存在: {self.config.YOLOV8S_MODEL_PATH}")
            
            if not os.path.exists(self.config.LICENSE_PLATE_MODEL_PATH):
                raise FileNotFoundError(f"车牌检测模型文件不存在: {self.config.LICENSE_PLATE_MODEL_PATH}")
            
            # 初始化YOLOv8s人脸检测模型
            logger.info("🔍 初始化YOLOv8s人脸检测模型...")
            self.face_model = YOLO(self.config.YOLOV8S_MODEL_PATH)
            logger.info("✅ YOLOv8s人脸检测模型初始化成功")
            
            # 初始化车牌检测模型
            logger.info("🚗 初始化车牌检测模型...")
            self.plate_model = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
            logger.info("✅ 车牌检测模型初始化成功")
            
            # 测试RetinaFace
            logger.info("🔍 测试RetinaFace模型...")
            self._test_retinaface()
            
            # 初始化OCR
            logger.info("📝 初始化EasyOCR模型...")
            self._initialize_ocr()
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def _test_retinaface(self):
        """测试RetinaFace模型"""
        try:
            # 创建测试图像
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = RetinaFace.detect_faces(test_img)
            logger.info("✅ RetinaFace模型测试成功")
        except Exception as e:
            logger.error(f"❌ RetinaFace测试失败: {e}")
            raise
    
    def _initialize_ocr(self):
        """初始化OCR模型"""
        try:
            gpu_enabled = 'cuda' in self.device
            self.ocr_reader = easyocr.Reader(
                ['ch_sim', 'en'],  # 支持中文简体和英文
                gpu=gpu_enabled
            )
            logger.info(f"✅ EasyOCR模型初始化成功 ({'GPU' if gpu_enabled else 'CPU'}模式)")
        except Exception as e:
            logger.error(f"❌ OCR模型初始化失败: {e}")
            self.ocr_reader = None
    
    def detect_faces_yolo(self, image_path: str) -> Tuple[int, List[Dict]]:
        """使用YOLOv8s检测人脸"""
        try:
            results = self.face_model(image_path, verbose=False, device=self.device)
            
            if not results or len(results) == 0:
                return 0, []
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return 0, []
            
            # 读取图像获取尺寸信息
            img = cv2.imread(image_path, self.config.IMAGE_READ_FLAGS)
            if img is None:
                return 0, []
            
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            
            face_detections = []
            close_up_faces = 0
            
            for box in result.boxes:
                try:
                    confidence = float(box.conf[0])
                    if confidence < self.config.MIN_FACE_CONFIDENCE_YOLO:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_area = face_width * face_height
                    
                    # 检查最小尺寸
                    if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                        continue
                    
                    if face_area < self.config.MIN_FACE_AREA:
                        continue
                    
                    # 计算面积比例判断是否为近景
                    area_ratio = face_area / img_area
                    is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                    
                    face_info = {
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'face_size': (float(face_width), float(face_height)),
                        'face_area': float(face_area),
                        'area_ratio': float(area_ratio),
                        'is_close_up': is_close_up
                    }
                    
                    face_detections.append(face_info)
                    
                    if is_close_up:
                        close_up_faces += 1
                
                except Exception as e:
                    logger.debug(f"分析YOLO检测框失败: {e}")
                    continue
            
            return close_up_faces, face_detections
            
        except Exception as e:
            logger.debug(f"YOLO人脸检测失败 {image_path}: {e}")
            return 0, []
    
    def calculate_yaw_angle(self, landmarks: Dict) -> float:
        """基于RetinaFace关键点计算yaw角度"""
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width < 10:
                return 90.0
            
            horizontal_offset = nose[0] - eye_center[0]
            normalized_offset = horizontal_offset / eye_width
            yaw_angle = abs(normalized_offset) * 60.0
            
            return yaw_angle
            
        except Exception as e:
            logger.debug(f"yaw角度计算失败: {e}")
            return 90.0
    
    def detect_frontal_faces_retina(self, image_path: str) -> Tuple[int, List[Dict]]:
        """使用RetinaFace检测正脸"""
        try:
            # 直接传递图片路径给RetinaFace
            detections = RetinaFace.detect_faces(image_path)
            
            if not isinstance(detections, dict) or len(detections) == 0:
                return 0, []
            
            # 读取图像信息
            img = cv2.imread(image_path, self.config.IMAGE_READ_FLAGS)
            if img is None:
                return 0, []
            
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            
            frontal_faces = []
            close_frontal_count = 0
            
            for face_key, face_data in detections.items():
                try:
                    confidence = face_data.get('score', 0.0)
                    if confidence < self.config.MIN_FACE_CONFIDENCE_RETINA:
                        continue
                    
                    facial_area = face_data['facial_area']
                    landmarks = face_data.get('landmarks', {})
                    
                    if not landmarks:
                        continue
                    
                    x1, y1, x2, y2 = facial_area
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_area = face_width * face_height
                    
                    # 检查最小尺寸
                    if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                        continue
                    
                    if face_area < self.config.MIN_FACE_AREA:
                        continue
                    
                    # 计算yaw角度
                    yaw_angle = self.calculate_yaw_angle(landmarks)
                    is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                    
                    # 判断是否为近景
                    area_ratio = face_area / img_area
                    is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                    
                    face_info = {
                        'confidence': confidence,
                        'yaw_angle': yaw_angle,
                        'is_frontal': is_frontal,
                        'facial_area': facial_area,
                        'face_size': (face_width, face_height),
                        'face_area': face_area,
                        'area_ratio': area_ratio,
                        'is_close_up': is_close_up,
                        'is_close_frontal': is_frontal and is_close_up
                    }
                    
                    frontal_faces.append(face_info)
                    
                    if is_frontal and is_close_up:
                        close_frontal_count += 1
                
                except Exception as e:
                    logger.debug(f"分析RetinaFace检测结果失败: {e}")
                    continue
            
            return close_frontal_count, frontal_faces
            
        except Exception as e:
            logger.debug(f"RetinaFace检测失败 {image_path}: {e}")
            return 0, []
    
    def detect_license_plates(self, image_path: str) -> Tuple[int, List[Dict]]:
        """检测车牌"""
        try:
            results = self.plate_model(image_path, verbose=False, device=self.device)
            
            if not results or len(results) == 0:
                return 0, []
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return 0, []
            
            clear_plates = []
            
            for box in result.boxes:
                try:
                    confidence = float(box.conf[0])
                    if confidence < self.config.MIN_PLATE_CONFIDENCE:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    
                    plate_info = {
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'plate_size': (float(plate_width), float(plate_height))
                    }
                    
                    clear_plates.append(plate_info)
                
                except Exception as e:
                    logger.debug(f"分析车牌失败: {e}")
                    continue
            
            return len(clear_plates), clear_plates
            
        except Exception as e:
            logger.debug(f"车牌检测失败 {image_path}: {e}")
            return 0, []
    
    def detect_text(self, image_path: str) -> Tuple[int, List[Dict]]:
        """检测文字"""
        try:
            if self.ocr_reader is None:
                return 0, []
            
            img = cv2.imread(image_path, self.config.IMAGE_READ_FLAGS)
            if img is None:
                return 0, []
            
            results = self.ocr_reader.readtext(img)
            
            if not results:
                return 0, []
            
            valid_texts = []
            
            for bbox, text, confidence in results:
                try:
                    confidence = float(confidence) if confidence is not None else 0.0
                    
                    if confidence < self.config.MIN_TEXT_CONFIDENCE:
                        continue
                    
                    cleaned_text = text.strip()
                    if len(cleaned_text) < 2:
                        continue
                    
                    text_info = {
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    valid_texts.append(text_info)
                
                except Exception as e:
                    logger.debug(f"分析文字失败: {e}")
                    continue
            
            return len(valid_texts), valid_texts
            
        except Exception as e:
            logger.debug(f"文字检测失败 {image_path}: {e}")
            return 0, []
    
    def classify_image(self, image_path: str) -> Tuple[str, Dict]:
        """分类图像"""
        start_time = time.time()
        
        try:
            filename = os.path.basename(image_path)
            
            # 检查文件是否存在和有效
            if not os.path.exists(image_path):
                return 'failed', {'filename': filename, 'error': '文件不存在'}
            
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return 'failed', {'filename': filename, 'error': '文件为空'}
            
            # 第一步：使用YOLOv8s检测人脸
            yolo_close_faces, yolo_face_details = self.detect_faces_yolo(image_path)
            
            # 第二步：使用RetinaFace检测正脸
            retina_close_frontal, retina_face_details = self.detect_frontal_faces_retina(image_path)
            
            # 判断是否有足够的近景正脸
            has_enough_faces = retina_close_frontal >= self.config.REQUIRED_CLOSE_FRONTAL_FACES
            
            plate_count = 0
            text_count = 0
            plate_details = []
            text_details = []
            
            if not has_enough_faces:
                # 如果近景正脸不够，检测车牌和文字
                plate_count, plate_details = self.detect_license_plates(image_path)
                text_count, text_details = self.detect_text(image_path)
            
            processing_time = time.time() - start_time
            
            # 创建分析结果
            analysis = {
                'filename': filename,
                'yolo_close_faces': yolo_close_faces,
                'retina_close_frontal_faces': retina_close_frontal,
                'license_plates': plate_count,
                'text_count': text_count,
                'yolo_face_details': yolo_face_details,
                'retina_face_details': retina_face_details,
                'plate_details': plate_details,
                'text_details': text_details,
                'processing_time': processing_time,
                'file_size': file_size,
                'timestamp': time.time()
            }
            
            # 决定分类
            if has_enough_faces:
                category = 'satisfied_4_faces'
                analysis['result'] = f'满足条件：有{retina_close_frontal}张近景正脸(>={self.config.REQUIRED_CLOSE_FRONTAL_FACES})'
            elif retina_close_frontal > 0 and (plate_count > 0 or text_count > 0):
                category = 'satisfied_with_plate'
                analysis['result'] = f'满足条件：有{retina_close_frontal}张近景正脸 + {plate_count}个车牌 + {text_count}个文字'
            elif yolo_close_faces > 0 or retina_close_frontal > 0:
                category = 'insufficient_faces'
                analysis['result'] = f'人脸不足：YOLO检测{yolo_close_faces}张近景人脸，RetinaFace检测{retina_close_frontal}张近景正脸'
            else:
                category = 'no_faces'
                analysis['result'] = '无人脸检测到'
            
            analysis['category'] = category
            
            return category, analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像分类失败 {image_path}: {e}")
            return 'failed', {
                'filename': os.path.basename(image_path), 
                'error': str(e),
                'processing_time': processing_time
            }
    
    def move_image_to_category(self, image_path: str, category: str) -> bool:
        """移动图像到对应分类目录，确保不压缩图片"""
        try:
            filename = os.path.basename(image_path)
            
            if category not in self.output_dirs:
                logger.error(f"❌ 未知分类: {category}")
                return False
            
            output_dir = self.output_dirs[category]
            output_path = os.path.join(output_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
            
            # 使用shutil.copy2保持原始文件质量和元数据
            shutil.copy2(image_path, output_path)
            
            # 验证复制是否成功
            if not os.path.exists(output_path):
                logger.error(f"❌ 文件复制失败: {output_path}")
                return False
            
            # 验证文件大小
            original_size = os.path.getsize(image_path)
            copied_size = os.path.getsize(output_path)
            if original_size != copied_size:
                logger.warning(f"⚠️  文件大小不匹配: 原始={original_size}, 复制={copied_size}")
            
            # 删除原文件
            os.remove(image_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动图像失败 {image_path}: {e}")
            return False
    
    def get_image_files(self) -> List[str]:
        """获取所有图像文件"""
        files = []
        input_path = Path(self.config.INPUT_DIR)
        
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {self.config.INPUT_DIR}")
            return []
        
        logger.info(f"🔍 扫描目录: {self.config.INPUT_DIR}")
        
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            files.extend(input_path.glob(pattern))
        
        image_files = sorted([str(f) for f in files if f.is_file()])
        logger.info(f"📊 找到 {len(image_files)} 个图像文件")
        
        return image_files
    
    def save_analysis_results(self):
        """保存分析结果"""
        try:
            analysis_dir = self.output_dirs['analysis']
            
            # 保存详细分析结果
            analysis_file = os.path.join(analysis_dir, "advanced_classification_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            # 保存统计摘要
            summary = {
                'total_processed': len(self.analysis_results),
                'statistics': self.stats.copy(),
                'processing_time': time.time() - self.start_time,
                'configuration': {
                    'required_close_frontal_faces': self.config.REQUIRED_CLOSE_FRONTAL_FACES,
                    'yaw_angle_threshold': self.config.YAW_ANGLE_THRESHOLD,
                    'close_up_face_ratio': self.config.CLOSE_UP_FACE_RATIO,
                    'min_face_area': self.config.MIN_FACE_AREA,
                },
                'timestamp': time.time()
            }
            
            summary_file = os.path.join(analysis_dir, "advanced_classification_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 分析结果已保存到: {analysis_file}")
            logger.info(f"📊 统计摘要已保存到: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存分析结果失败: {e}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        processing_time = time.time() - self.start_time
        total_processed = sum(self.stats.values())
        
        logger.info("="*80)
        logger.info("🎉 高级人脸车牌检测分类完成！最终统计:")
        logger.info(f"✅ 4张近景正脸满足条件: {self.stats['satisfied_4_faces']:,}")
        logger.info(f"✅ 车牌/文字满足条件: {self.stats['satisfied_with_plate']:,}")
        logger.info(f"❌ 人脸数量不足: {self.stats['insufficient_faces']:,}")
        logger.info(f"❌ 无人脸: {self.stats['no_faces']:,}")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,}")
        logger.info(f"📊 总处理数量: {total_processed:,}")
        logger.info(f"⏰ 总耗时: {processing_time:.1f}秒")
        
        if total_processed > 0:
            avg_speed = total_processed / processing_time
            logger.info(f"🚀 平均速度: {avg_speed:.1f} 张/秒")
            
            success_rate = ((self.stats['satisfied_4_faces'] + self.stats['satisfied_with_plate']) / total_processed) * 100
            logger.info(f"📈 满足条件比例: {success_rate:.1f}%")
        
        # 显示各目录文件数量
        logger.info("\n📂 各分类目录统计:")
        categories = [
            ("4张近景正脸满足", self.output_dirs['satisfied_4_faces']),
            ("车牌文字满足", self.output_dirs['satisfied_with_plate']),
            ("人脸数量不足", self.output_dirs['insufficient_faces']),
            ("无人脸", self.output_dirs['no_faces'])
        ]
        
        for name, dir_path in categories:
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) 
                           if f.lower().endswith(tuple(self.config.SUPPORTED_FORMATS))])
                logger.info(f"  📁 {name}: {count} 张图片")
        
        logger.info("="*80)
    
    def run(self):
        """运行检测器"""
        logger.info("🚀 启动高级人脸车牌检测器...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📁 输出目录: {self.config.OUTPUT_BASE_DIR}")
        logger.info(f"💻 计算设备: {self.device}")
        logger.info(f"🎯 需要近景正脸数量: {self.config.REQUIRED_CLOSE_FRONTAL_FACES}")
        logger.info(f"📐 yaw角度阈值: {self.config.YAW_ANGLE_THRESHOLD}°")
        logger.info(f"📏 近景面积比例阈值: {self.config.CLOSE_UP_FACE_RATIO}")
        logger.info(f"🔍 YOLO人脸置信度阈值: {self.config.MIN_FACE_CONFIDENCE_YOLO}")
        logger.info(f"🔍 RetinaFace置信度阈值: {self.config.MIN_FACE_CONFIDENCE_RETINA}")
        
        # 获取图像文件
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("❌ 未找到任何图像文件")
            return
        
        total_files = len(image_files)
        logger.info(f"📊 找到 {total_files} 张图片待处理")
        
        # 开始处理
        try:
            start_time = time.time()
            failed_files = []
            
            for i, image_path in enumerate(image_files):
                try:
                    # 分类图像
                    category, analysis = self.classify_image(image_path)
                    
                    if category != 'failed':
                        # 移动到对应目录
                        if self.move_image_to_category(image_path, category):
                            self.stats[category] += 1
                        else:
                            self.stats['failed'] += 1
                            analysis['move_failed'] = True
                            failed_files.append(image_path)
                    else:
                        self.stats['failed'] += 1
                        failed_files.append(image_path)
                    
                    # 保存分析结果
                    self.analysis_results.append(analysis)
                    
                    # 进度显示
                    if (i + 1) % self.config.PROGRESS_UPDATE_FREQUENCY == 0 or i == total_files - 1:
                        progress = (i + 1) / total_files * 100
                        elapsed = time.time() - start_time
                        
                        if i > 0:
                            avg_time = elapsed / (i + 1)
                            remaining_files = total_files - (i + 1)
                            eta = remaining_files * avg_time
                            
                            eta_str = f"{eta/60:.1f}分钟" if eta > 60 else f"{eta:.0f}秒"
                            
                            print(f"\r📊 进度: {i+1}/{total_files} ({progress:.1f}%) "
                                  f"| 已用时: {elapsed/60:.1f}分钟 "
                                  f"| 预计剩余: {eta_str} "
                                  f"| 速度: {(i+1)/elapsed:.1f}张/秒", end='', flush=True)
                        else:
                            print(f"\r📊 进度: {i+1}/{total_files} ({progress:.1f}%)", end='', flush=True)
                    
                    # 定期内存清理
                    if (i + 1) % self.config.GC_FREQUENCY == 0:
                        gc.collect()
                        if 'cuda' in self.device:
                            torch.cuda.empty_cache()
                
                except KeyboardInterrupt:
                    logger.info("\n⚡ 用户中断操作")
                    break
                except Exception as e:
                    logger.error(f"❌ 处理图像失败 {image_path}: {e}")
                    self.stats['failed'] += 1
                    failed_files.append(image_path)
            
            # 完成后换行
            print()
            
            # 显示失败文件
            if failed_files:
                logger.warning(f"⚠️  失败文件数: {len(failed_files)}")
                if len(failed_files) <= 10:
                    for failed_file in failed_files:
                        logger.warning(f"  - {os.path.basename(failed_file)}")
                else:
                    logger.warning(f"  显示前10个: {[os.path.basename(f) for f in failed_files[:10]]}")
        
        finally:
            # 保存结果和统计
            self.save_analysis_results()
            self.print_final_statistics()

def main():
    """主函数"""
    try:
        # 创建配置
        config = Config()
        
        # 检查输入目录
        if not os.path.exists(config.INPUT_DIR):
            logger.error(f"❌ 输入目录不存在: {config.INPUT_DIR}")
            return
        
        # 检查模型文件
        if not os.path.exists(config.YOLOV8S_MODEL_PATH):
            logger.error(f"❌ YOLOv8s模型不存在: {config.YOLOV8S_MODEL_PATH}")
            return
        
        if not os.path.exists(config.LICENSE_PLATE_MODEL_PATH):
            logger.error(f"❌ 车牌检测模型不存在: {config.LICENSE_PLATE_MODEL_PATH}")
            return
        
        # 创建检测器并运行
        detector = AdvancedFacePlateDetector(config)
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
