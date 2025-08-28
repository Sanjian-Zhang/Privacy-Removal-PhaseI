#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正脸和车牌检测分类器 - 高性能优化版本
主要优化点：
1. 多线程/多进程并行处理
2. 批量图片预处理
3. 内存池管理
4. 图片预加载和缓存
5. 优化的GPU内存管理
6. 减少重复计算
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import multiprocessing as mp
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# 配置环境变量 - 必须在导入深度学习库之前设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 启用GPU 0和1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # 减少TensorFlow日志输出
os.environ['OMP_NUM_THREADS'] = '4'          # 限制OpenMP线程数
os.environ['OPENBLAS_NUM_THREADS'] = '4'     # 限制OpenBLAS线程数

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_plate_classifier_optimized.log', encoding='utf-8')
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
    
    # 检查torch（用于GPU检查）
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

class OptimizedConfig:
    """优化配置类"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames84'
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames84'
    PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 新计分系统阈值
    SCORE_THRESHOLD = 5                # 总分阈值（>5分符合要求）
    CLEAR_FACE_SCORE = 2              # 清晰正脸得分
    CLEAR_PLATE_SCORE = 2             # 清晰车牌得分
    TEXT_RECOGNITION_SCORE = 2        # 文字识别得分（有文字即得分）
    
    # 检测阈值
    YAW_ANGLE_THRESHOLD = 35.0        # yaw角度阈值
    MIN_FACE_CONFIDENCE = 0.8         # 最小人脸置信度
    MIN_PLATE_CONFIDENCE = 0.5        # 最小车牌置信度
    MIN_FACE_SIZE = 60                # 最小人脸尺寸
    MIN_PLATE_SIZE = 50               # 最小车牌尺寸
    MIN_FACE_CLARITY_SCORE = 30.0     # 最小清晰度分数
    MAX_FACE_DISTANCE_RATIO = 0.3     # 最大距离比例
    FACE_AREA_THRESHOLD = 3600        # 人脸面积阈值
    MIN_TEXT_CONFIDENCE = 0.5         # 最小文字置信度
    MIN_TEXT_LENGTH = 3               # 最小文字长度
    
    # 图像处理参数
    MAX_IMAGE_SIZE = (1280, 720)
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 性能优化参数
    MAX_WORKERS = min(8, mp.cpu_count())        # 最大工作线程数
    BATCH_SIZE = 32                             # 批处理大小
    PREFETCH_BUFFER_SIZE = 100                  # 预取缓冲区大小
    IMAGE_CACHE_SIZE = 1000                     # 图像缓存大小
    GC_FREQUENCY = 50                           # 垃圾回收频率
    PROGRESS_UPDATE_FREQUENCY = 20              # 进度更新频率
    ENABLE_IMAGE_CACHE = True                   # 启用图像缓存
    ENABLE_PARALLEL_PROCESSING = True           # 启用并行处理
    USE_PROCESS_POOL = False                    # 使用进程池（默认使用线程池）
    
    # GPU配置（启用GPU加速）
    USE_GPU = True
    GPU_DEVICE_ID = 1                 # 使用GPU 1（GPU 0正在被占用）
    ENABLE_TORCH_OPTIMIZATION = True
    GPU_MEMORY_FRACTION = 0.8         # GPU内存使用比例
    
    # 图像预处理优化 - 修改为保护图像质量
    ENABLE_IMAGE_PREPROCESSING = False  # 禁用图像预处理以保持原始质量
    RESIZE_FOR_SPEED = False           # 禁用为速度调整图像大小
    SPEED_RESIZE_SIZE = (640, 480)     # 速度优化时的图像大小（已禁用）
    
    # 图像质量保护设置
    PRESERVE_IMAGE_QUALITY = True      # 确保不压缩图片
    IMAGE_READ_FLAGS = cv2.IMREAD_COLOR  # 使用高质量读取标志
    JPEG_QUALITY = 100                 # JPEG保存质量（如果需要保存）
    PNG_COMPRESSION = 0                # PNG压缩级别（0=无压缩）
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'qualified': os.path.join(cls.OUTPUT_BASE_DIR, "qualified"),
            'qualified_1_4_faces': os.path.join(cls.OUTPUT_BASE_DIR, "qualified", "1-4张人脸"),
            'qualified_5_8_faces': os.path.join(cls.OUTPUT_BASE_DIR, "qualified", "5-8张人脸"),
            'qualified_9_plus_faces': os.path.join(cls.OUTPUT_BASE_DIR, "qualified", "9张人脸以上"),
            'insufficient_score': os.path.join(cls.OUTPUT_BASE_DIR, "insufficient_score"),
            'no_content': os.path.join(cls.OUTPUT_BASE_DIR, "no_content"),
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis")
        }

class ImageCache:
    """线程安全的图像缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
    
    def get(self, path: str) -> Optional[np.ndarray]:
        """获取缓存的图像"""
        with self.lock:
            if path in self.cache:
                # 更新访问顺序
                self.access_order.remove(path)
                self.access_order.append(path)
                return self.cache[path].copy()
            return None
    
    def put(self, path: str, image: np.ndarray):
        """添加图像到缓存"""
        with self.lock:
            # 如果缓存已满，删除最旧的
            if len(self.cache) >= self.max_size:
                oldest_path = self.access_order.pop(0)
                del self.cache[oldest_path]
            
            self.cache[path] = image.copy()
            if path in self.access_order:
                self.access_order.remove(path)
            self.access_order.append(path)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class OptimizedImageProcessor:
    """优化的图像处理器"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.image_cache = ImageCache(config.IMAGE_CACHE_SIZE) if config.ENABLE_IMAGE_CACHE else None
    
    def load_image_optimized(self, image_path: str) -> Optional[np.ndarray]:
        """优化的图像加载，保持原始质量"""
        try:
            # 检查缓存
            if self.image_cache:
                cached_img = self.image_cache.get(image_path)
                if cached_img is not None:
                    return cached_img
            
            # 使用高质量标志加载图像
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            # 跳过图像预处理以保持原始质量
            # 图像预处理优化已禁用以保护图像质量
            
            # 添加到缓存
            if self.image_cache:
                self.image_cache.put(image_path, img)
            
            return img
            
        except Exception as e:
            logger.debug(f"图像加载失败 {image_path}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理 - 已禁用以保护图像质量"""
        try:
            # 预处理已禁用以保持原始图像质量
            # 直接返回原始图像，不进行任何处理
            return image
            
        except Exception as e:
            logger.debug(f"图像预处理失败: {e}")
            return image

class ModelManager:
    """模型管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: OptimizedConfig):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: OptimizedConfig):
        if self._initialized:
            return
        
        self.config = config
        self.device = self._setup_device()
        self._initialize_models()
        self._initialized = True
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        try:
            if self.config.USE_GPU and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                if self.config.GPU_DEVICE_ID < device_count:
                    device = f'cuda:{self.config.GPU_DEVICE_ID}'
                    gpu_name = torch.cuda.get_device_name(self.config.GPU_DEVICE_ID)
                    gpu_memory = torch.cuda.get_device_properties(self.config.GPU_DEVICE_ID).total_memory / 1024**3
                    logger.info(f"🚀 GPU加速已启用: {gpu_name} (设备 {self.config.GPU_DEVICE_ID})")
                    logger.info(f"🔥 GPU显存: {gpu_memory:.1f} GB")
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
                    # 设置PyTorch优化
                    if self.config.ENABLE_TORCH_OPTIMIZATION:
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                        logger.info("⚡ PyTorch优化已启用")
                    
                    return device
                else:
                    logger.warning(f"⚠️  指定的GPU设备ID {self.config.GPU_DEVICE_ID} 超出范围，共有 {device_count} 个GPU")
                    return 'cpu'
            else:
                if not torch.cuda.is_available():
                    logger.info("💻 未检测到CUDA设备，使用CPU模式")
                else:
                    logger.info("💻 GPU加速已禁用，使用CPU模式")
                return 'cpu'
                
        except Exception as e:
            logger.error(f"❌ 设备设置失败: {e}")
            return 'cpu'
    
    def _initialize_models(self):
        """初始化检测模型"""
        try:
            # 配置TensorFlow GPU使用
            try:
                import tensorflow as tf
                
                if 'cuda' in self.device:
                    # 配置GPU内存增长
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                                # 设置内存限制
                                memory_limit = int(tf.config.experimental.get_device_details(gpu)['device_name'])
                                tf.config.experimental.set_memory_growth(gpu, True)
                            logger.info("🚀 TensorFlow已配置为GPU模式（内存增长）")
                        except RuntimeError as e:
                            logger.warning(f"⚠️  GPU内存增长配置失败: {e}")
                else:
                    tf.config.set_visible_devices([], 'GPU')
                    logger.info("🔧 TensorFlow已配置为CPU模式")
            except ImportError:
                logger.info("ℹ️  未检测到TensorFlow")
            
            # 初始化车牌检测模型
            logger.info("🚗 初始化车牌检测模型...")
            self._initialize_plate_model()
            
            # 初始化OCR模型
            logger.info("📝 初始化EasyOCR模型...")
            self._initialize_ocr()
            
            # 初始化RetinaFace（延迟初始化，首次使用时加载）
            self.retinaface_initialized = False
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def _initialize_plate_model(self):
        """初始化车牌检测模型"""
        if not os.path.exists(self.config.PLATE_MODEL_PATH):
            raise FileNotFoundError(f"车牌检测模型文件不存在: {self.config.PLATE_MODEL_PATH}")
        
        self.plate_model = YOLO(self.config.PLATE_MODEL_PATH)
        
        if 'cuda' in self.device:
            logger.info(f"✅ 车牌检测模型初始化成功（GPU模式 - {self.device}）")
        else:
            logger.info("✅ 车牌检测模型初始化成功（CPU模式）")
    
    def _initialize_ocr(self):
        """初始化OCR模型"""
        try:
            # 设置EasyOCR（根据设备配置）
            gpu_enabled = 'cuda' in self.device
            self.ocr_reader = easyocr.Reader(
                ['ch_sim', 'en'],  # 支持中文简体和英文
                gpu=gpu_enabled    # 根据设备配置启用/禁用GPU
            )
            
            if gpu_enabled:
                logger.info(f"✅ EasyOCR模型初始化成功 (GPU模式 - {self.device})")
            else:
                logger.info("✅ EasyOCR模型初始化成功 (CPU模式)")
                
        except Exception as e:
            logger.error(f"❌ OCR模型初始化失败: {e}")
            self.ocr_reader = None
    
    def _ensure_retinaface_initialized(self):
        """确保RetinaFace已初始化"""
        if not self.retinaface_initialized:
            try:
                # 创建测试图像
                test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
                RetinaFace.detect_faces(test_img)
                self.retinaface_initialized = True
                logger.info("✅ RetinaFace模型初始化成功")
            except Exception as e:
                logger.error(f"❌ RetinaFace初始化失败: {e}")
                raise

class OptimizedFacePlateClassifier:
    """优化的正脸和车牌检测分类器"""
    
    def __init__(self, config: Optional[OptimizedConfig] = None):
        """初始化分类器"""
        self.config = config or OptimizedConfig()
        self.start_time = time.time()
        
        # 统计信息
        self.stats = {
            'qualified': 0,           # 符合条件(总分>5)
            'qualified_1_4_faces': 0, # 符合条件且1-4张人脸
            'qualified_5_8_faces': 0, # 符合条件且5-8张人脸
            'qualified_9_plus_faces': 0, # 符合条件且9张人脸以上
            'insufficient_score': 0,  # 分数不够
            'no_content': 0,          # 无任何有效内容
            'failed': 0               # 处理失败
        }
        
        # 详细分析结果
        self.analysis_results = []
        self.analysis_lock = threading.Lock()
        
        # 获取输出目录
        self.output_dirs = self.config.get_output_dirs()
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 初始化组件
        self.image_processor = OptimizedImageProcessor(self.config)
        self.model_manager = ModelManager(self.config)
        
        logger.info("🚀 正脸和车牌检测分类器初始化完成（高性能优化版本）")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        for name, dir_path in self.output_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ 创建目录失败 {dir_path}: {e}")
                raise
    
    def calculate_face_clarity(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """计算人脸区域的清晰度 - 优化版本"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            face_region = image[y1:y2, x1:x2]
            if face_region.size == 0:
                return 0.0
            
            # 转换为灰度图
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            # 快速清晰度计算
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            return float(laplacian_var)
            
        except Exception as e:
            logger.debug(f"清晰度计算失败: {e}")
            return 0.0
    
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
    
    def is_face_clear_and_close(self, image: np.ndarray, bbox: Tuple[int, int, int, int], img_size: Tuple[int, int]) -> Tuple[bool, Dict]:
        """判断人脸是否清晰且距离合适"""
        try:
            clarity_score = self.calculate_face_clarity(image, bbox)
            
            x1, y1, x2, y2 = bbox
            face_area = (x2 - x1) * (y2 - y1)
            img_width, img_height = img_size
            img_area = img_width * img_height
            distance_score = face_area / img_area
            
            is_clear = clarity_score >= self.config.MIN_FACE_CLARITY_SCORE
            is_close = distance_score >= self.config.MAX_FACE_DISTANCE_RATIO
            is_large_enough = face_area >= self.config.FACE_AREA_THRESHOLD
            
            is_good_quality = is_clear and (is_close or is_large_enough)
            
            quality_info = {
                'clarity_score': clarity_score,
                'distance_score': distance_score,
                'face_area': face_area,
                'is_clear': is_clear,
                'is_close': is_close,
                'is_large_enough': is_large_enough,
                'is_good_quality': is_good_quality
            }
            
            return is_good_quality, quality_info
            
        except Exception as e:
            logger.debug(f"质量评估失败: {e}")
            return False, {'error': str(e)}
    
    def detect_faces_batch(self, image_batch: List[Tuple[str, np.ndarray]]) -> List[Tuple[int, List[Dict]]]:
        """批量人脸检测 - 针对多个图像"""
        results = []
        
        # 确保RetinaFace已初始化
        self.model_manager._ensure_retinaface_initialized()
        
        for image_path, img in image_batch:
            try:
                detections = RetinaFace.detect_faces(img)
                
                if not isinstance(detections, dict) or len(detections) == 0:
                    results.append((0, []))
                    continue
                
                img_height, img_width = img.shape[:2]
                img_size = (img_width, img_height)
                
                clear_frontal_faces = []
                
                for face_key, face_data in detections.items():
                    try:
                        confidence = face_data.get('score', 0.0)
                        if confidence < self.config.MIN_FACE_CONFIDENCE:
                            continue
                        
                        facial_area = face_data['facial_area']
                        landmarks = face_data.get('landmarks', {})
                        
                        if not landmarks:
                            continue
                        
                        x1, y1, x2, y2 = facial_area
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                            continue
                        
                        # 检查人脸清晰度和距离
                        is_good_quality, quality_info = self.is_face_clear_and_close(img, facial_area, img_size)
                        
                        if not is_good_quality:
                            continue
                        
                        # 计算yaw角度
                        yaw_angle = self.calculate_yaw_angle(landmarks)
                        
                        # 判断是否为正脸
                        is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                        
                        if is_frontal:  # 只记录清晰的正脸
                            face_info = {
                                'confidence': confidence,
                                'yaw_angle': yaw_angle,
                                'is_frontal': is_frontal,
                                'facial_area': facial_area,
                                'face_size': (face_width, face_height),
                                'quality_info': quality_info
                            }
                            
                            clear_frontal_faces.append(face_info)
                    
                    except Exception as e:
                        logger.debug(f"分析人脸失败: {e}")
                        continue
                
                results.append((len(clear_frontal_faces), clear_frontal_faces))
                
            except Exception as e:
                logger.debug(f"人脸检测失败 {image_path}: {e}")
                results.append((0, []))
        
        return results
    
    def detect_plates_batch(self, image_batch: List[Tuple[str, np.ndarray]]) -> List[Tuple[int, List[Dict]]]:
        """批量车牌检测"""
        results = []
        
        # 准备图像路径列表
        image_paths = [path for path, _ in image_batch]
        
        try:
            # 批量推理
            batch_results = self.model_manager.plate_model(image_paths, verbose=False, device=self.model_manager.device)
            
            for i, (image_path, img) in enumerate(image_batch):
                try:
                    if i >= len(batch_results):
                        results.append((0, []))
                        continue
                    
                    result = batch_results[i]
                    
                    if result.boxes is None or len(result.boxes) == 0:
                        results.append((0, []))
                        continue
                    
                    clear_plates = []
                    
                    for box in result.boxes:
                        try:
                            confidence = float(box.conf[0])
                            if confidence < self.config.MIN_PLATE_CONFIDENCE:
                                continue
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            plate_width = x2 - x1
                            plate_height = y2 - y1
                            
                            if min(plate_width, plate_height) < self.config.MIN_PLATE_SIZE:
                                continue
                            
                            plate_info = {
                                'confidence': confidence,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'plate_size': (float(plate_width), float(plate_height))
                            }
                            
                            clear_plates.append(plate_info)
                        
                        except Exception as e:
                            logger.debug(f"分析车牌失败: {e}")
                            continue
                    
                    results.append((len(clear_plates), clear_plates))
                
                except Exception as e:
                    logger.debug(f"车牌检测失败 {image_path}: {e}")
                    results.append((0, []))
        
        except Exception as e:
            logger.debug(f"批量车牌检测失败: {e}")
            # 如果批量处理失败，返回空结果
            results = [(0, [])] * len(image_batch)
        
        return results
    
    def detect_text_batch(self, image_batch: List[Tuple[str, np.ndarray]]) -> List[Tuple[int, List[Dict]]]:
        """批量文字检测"""
        results = []
        
        if self.model_manager.ocr_reader is None:
            return [(0, [])] * len(image_batch)
        
        for image_path, img in image_batch:
            try:
                ocr_results = self.model_manager.ocr_reader.readtext(img)
                
                if not ocr_results:
                    results.append((0, []))
                    continue
                
                valid_texts = []
                
                for bbox, text, confidence in ocr_results:
                    try:
                        # 确保confidence是float类型
                        confidence = float(confidence) if confidence is not None else 0.0
                        
                        if confidence < self.config.MIN_TEXT_CONFIDENCE:
                            continue
                        
                        cleaned_text = text.strip()
                        if len(cleaned_text) < self.config.MIN_TEXT_LENGTH:
                            continue
                        
                        # 过滤掉只包含符号的文字
                        if cleaned_text.replace(' ', '').replace('.', '').replace('-', '').replace('_', ''):
                            text_info = {
                                'text': cleaned_text,
                                'confidence': confidence,
                                'bbox': bbox
                            }
                            valid_texts.append(text_info)
                    
                    except Exception as e:
                        logger.debug(f"分析文字失败: {e}")
                        continue
                
                results.append((len(valid_texts), valid_texts))
                
            except Exception as e:
                logger.debug(f"文字检测失败 {image_path}: {e}")
                results.append((0, []))
        
        return results
    
    def process_image_batch(self, image_paths: List[str]) -> List[Tuple[str, str, Dict]]:
        """批量处理图像"""
        try:
            # 加载图像批次
            image_batch = []
            for path in image_paths:
                img = self.image_processor.load_image_optimized(path)
                if img is not None:
                    image_batch.append((path, img))
                else:
                    # 如果图像加载失败，跳过
                    continue
            
            if not image_batch:
                return []
            
            # 批量检测
            face_results = self.detect_faces_batch(image_batch)
            plate_results = self.detect_plates_batch(image_batch)
            text_results = self.detect_text_batch(image_batch)
            
            # 整合结果
            batch_results = []
            for i, (image_path, img) in enumerate(image_batch):
                try:
                    filename = os.path.basename(image_path)
                    
                    # 获取检测结果
                    frontal_count, face_details = face_results[i] if i < len(face_results) else (0, [])
                    plate_count, plate_details = plate_results[i] if i < len(plate_results) else (0, [])
                    text_count, text_details = text_results[i] if i < len(text_results) else (0, [])
                    
                    # 新计分系统
                    score = 0
                    score_details = []
                    
                    # 清晰正脸：每张2分
                    if frontal_count > 0:
                        face_score = frontal_count * self.config.CLEAR_FACE_SCORE
                        score += face_score
                        score_details.append(f"清晰正脸 {frontal_count} 张 × {self.config.CLEAR_FACE_SCORE} = {face_score} 分")
                    
                    # 清晰车牌：每张2分
                    if plate_count > 0:
                        plate_score = plate_count * self.config.CLEAR_PLATE_SCORE
                        score += plate_score
                        score_details.append(f"清晰车牌 {plate_count} 张 × {self.config.CLEAR_PLATE_SCORE} = {plate_score} 分")
                    
                    # 可识别文字：有文字就得分
                    if text_count > 0:
                        text_score = self.config.TEXT_RECOGNITION_SCORE
                        score += text_score
                        score_details.append(f"可识别文字 {text_count} 个字段 = {text_score} 分")
                    
                    # 判断是否符合要求（总分>5）
                    meets_requirements = score > self.config.SCORE_THRESHOLD
                    
                    # 创建分析结果
                    analysis = {
                        'filename': filename,
                        'frontal_faces': frontal_count,
                        'license_plates': plate_count,
                        'text_count': text_count,
                        'total_score': score,
                        'score_details': score_details,
                        'meets_requirements': meets_requirements,
                        'score_threshold': self.config.SCORE_THRESHOLD,
                        'face_details': face_details,
                        'plate_details': plate_details,
                        'text_details': text_details,
                        'timestamp': time.time()
                    }
                    
                    # 分类逻辑
                    if meets_requirements:
                        # 根据人脸数量进一步分类
                        face_category = self.get_face_count_category(frontal_count)
                        category = face_category
                        analysis['qualification_reason'] = f'总分 {score} 分 > {self.config.SCORE_THRESHOLD} 分，符合要求'
                        analysis['face_count_category'] = face_category
                        
                        # 添加人脸数量说明
                        if frontal_count == 0:
                            analysis['face_count_description'] = '无人脸但其他条件满足'
                        elif 1 <= frontal_count <= 4:
                            analysis['face_count_description'] = f'{frontal_count}张人脸 (1-4张)'
                        elif 5 <= frontal_count <= 8:
                            analysis['face_count_description'] = f'{frontal_count}张人脸 (5-8张)'
                        else:
                            analysis['face_count_description'] = f'{frontal_count}张人脸 (9张以上)'
                    else:
                        if score == 0:
                            category = 'no_content'
                            analysis['reject_reason'] = f'总分 {score} 分，无任何有效内容'
                        else:
                            category = 'insufficient_score'
                            analysis['reject_reason'] = f'总分 {score} 分 ≤ {self.config.SCORE_THRESHOLD} 分，不符合要求'
                    
                    analysis['category'] = category
                    
                    batch_results.append((image_path, category, analysis))
                
                except Exception as e:
                    logger.error(f"❌ 图像分析失败 {image_path}: {e}")
                    error_analysis = {'filename': os.path.basename(image_path), 'error': str(e)}
                    batch_results.append((image_path, 'failed', error_analysis))
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ 批量处理失败: {e}")
            return []
    
    def get_face_count_category(self, face_count: int) -> str:
        """根据人脸数量确定分类"""
        if 1 <= face_count <= 4:
            return 'qualified_1_4_faces'
        elif 5 <= face_count <= 8:
            return 'qualified_5_8_faces'
        elif face_count >= 9:
            return 'qualified_9_plus_faces'
        else:
            return 'qualified'  # 默认分类（0张人脸但其他条件满足）
    
    def move_image_to_category(self, image_path: str, category: str) -> bool:
        """移动图像到对应分类目录"""
        try:
            filename = os.path.basename(image_path)
            
            if category not in self.output_dirs:
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
            
            shutil.move(image_path, output_path)
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
            analysis_file = os.path.join(analysis_dir, "classification_analysis_optimized.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            # 保存统计摘要
            summary = {
                'total_processed': len(self.analysis_results),
                'statistics': self.stats.copy(),
                'processing_time': time.time() - self.start_time,
                'optimization_config': {
                    'max_workers': self.config.MAX_WORKERS,
                    'batch_size': self.config.BATCH_SIZE,
                    'enable_parallel_processing': self.config.ENABLE_PARALLEL_PROCESSING,
                    'enable_image_cache': self.config.ENABLE_IMAGE_CACHE,
                    'use_gpu': self.config.USE_GPU,
                    'gpu_device_id': self.config.GPU_DEVICE_ID
                },
                'scoring_system': {
                    'clear_face_score': self.config.CLEAR_FACE_SCORE,
                    'clear_plate_score': self.config.CLEAR_PLATE_SCORE,
                    'text_recognition_score': self.config.TEXT_RECOGNITION_SCORE,
                    'score_threshold': self.config.SCORE_THRESHOLD
                },
                'timestamp': time.time()
            }
            
            summary_file = os.path.join(analysis_dir, "classification_summary_optimized.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 分析结果已保存到: {analysis_file}")
            logger.info(f"📊 统计摘要已保存到: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存分析结果失败: {e}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        processing_time = time.time() - self.start_time
        total_processed = (self.stats['qualified'] + self.stats['insufficient_score'] + 
                          self.stats['no_content'] + self.stats['failed'])
        
        logger.info("="*80)
        logger.info("🎉 正脸和车牌分类完成！最终统计（高性能优化版本）:")
        logger.info(f"⚡ 性能配置:")
        logger.info(f"  - 最大工作线程: {self.config.MAX_WORKERS}")
        logger.info(f"  - 批处理大小: {self.config.BATCH_SIZE}")
        logger.info(f"  - 并行处理: {'启用' if self.config.ENABLE_PARALLEL_PROCESSING else '禁用'}")
        logger.info(f"  - 图像缓存: {'启用' if self.config.ENABLE_IMAGE_CACHE else '禁用'}")
        logger.info(f"  - GPU加速: {'启用' if self.config.USE_GPU else '禁用'}")
        logger.info(f"  - 计算设备: {self.model_manager.device}")
        logger.info(f"✅ 符合条件总计(>{self.config.SCORE_THRESHOLD}分): {self.stats['qualified']:,}")
        logger.info(f"  📸 1-4张人脸: {self.stats['qualified_1_4_faces']:,}")
        logger.info(f"  👥 5-8张人脸: {self.stats['qualified_5_8_faces']:,}")
        logger.info(f"  👨‍👩‍👧‍👦 9张人脸以上: {self.stats['qualified_9_plus_faces']:,}")
        logger.info(f"❌ 分数不够(≤{self.config.SCORE_THRESHOLD}分): {self.stats['insufficient_score']:,}")
        logger.info(f"❌ 无任何内容: {self.stats['no_content']:,}")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,}")
        logger.info(f"📊 总处理数量: {total_processed:,}")
        logger.info(f"⏰ 总耗时: {processing_time:.1f}秒")
        
        if total_processed > 0:
            avg_speed = total_processed / processing_time
            logger.info(f"🚀 平均速度: {avg_speed:.1f} 张/秒")
            
            success_rate = (self.stats['qualified'] / total_processed) * 100
            logger.info(f"📈 符合条件比例: {success_rate:.1f}%")
        
        logger.info("="*80)
    
    def worker_function(self, image_batch: List[str]) -> List[Tuple[str, str, Dict]]:
        """工作线程函数"""
        return self.process_image_batch(image_batch)
    
    def run(self):
        """运行分类器"""
        logger.info("🚀 启动正脸和车牌检测分类器（高性能优化版本）...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📁 输出目录: {self.config.OUTPUT_BASE_DIR}")
        logger.info(f"💻 计算设备: {self.model_manager.device}")
        logger.info(f"⚡ 性能配置:")
        logger.info(f"  - 最大工作线程: {self.config.MAX_WORKERS}")
        logger.info(f"  - 批处理大小: {self.config.BATCH_SIZE}")
        logger.info(f"  - 并行处理: {'启用' if self.config.ENABLE_PARALLEL_PROCESSING else '禁用'}")
        logger.info(f"  - 图像缓存: {'启用' if self.config.ENABLE_IMAGE_CACHE else '禁用'}")
        
        # 获取图像文件
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("❌ 未找到任何图像文件")
            return
        
        try:
            if self.config.ENABLE_PARALLEL_PROCESSING and len(image_files) > self.config.BATCH_SIZE:
                self._run_parallel(image_files)
            else:
                self._run_sequential(image_files)
        
        finally:
            # 保存结果和统计
            self.save_analysis_results()
            self.print_final_statistics()
            
            # 清理缓存
            if self.image_processor.image_cache:
                self.image_processor.image_cache.clear()
            
            # 清理GPU内存
            if 'cuda' in self.model_manager.device:
                torch.cuda.empty_cache()
    
    def _run_sequential(self, image_files: List[str]):
        """顺序处理模式"""
        logger.info("🔄 使用顺序处理模式...")
        
        # 分批处理
        total_batches = (len(image_files) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.BATCH_SIZE
            end_idx = min(start_idx + self.config.BATCH_SIZE, len(image_files))
            batch = image_files[start_idx:end_idx]
            
            # 处理批次
            batch_results = self.process_image_batch(batch)
            self._process_batch_results(batch_results)
            
            # 显示进度
            progress = (batch_idx + 1) / total_batches * 100
            processed = end_idx
            total = len(image_files)
            print(f"\r进度: {processed}/{total} ({progress:.1f}%) - 批次 {batch_idx + 1}/{total_batches}", 
                  end='', flush=True)
            
            # 定期垃圾回收
            if batch_idx % 10 == 0:
                gc.collect()
                if 'cuda' in self.model_manager.device:
                    torch.cuda.empty_cache()
        
        print()  # 换行
    
    def _run_parallel(self, image_files: List[str]):
        """并行处理模式"""
        logger.info(f"🚀 使用并行处理模式 ({self.config.MAX_WORKERS} 个工作线程)...")
        
        # 分批次
        batches = []
        for i in range(0, len(image_files), self.config.BATCH_SIZE):
            batches.append(image_files[i:i + self.config.BATCH_SIZE])
        
        logger.info(f"📦 分为 {len(batches)} 个批次，每批次最多 {self.config.BATCH_SIZE} 张图片")
        
        processed_count = 0
        
        # 使用线程池
        if self.config.USE_PROCESS_POOL:
            executor_class = ProcessPoolExecutor
            logger.info("🔧 使用进程池处理")
        else:
            executor_class = ThreadPoolExecutor
            logger.info("🔧 使用线程池处理")
        
        with executor_class(max_workers=self.config.MAX_WORKERS) as executor:
            # 提交任务
            future_to_batch = {executor.submit(self.worker_function, batch): batch for batch in batches}
            
            # 处理完成的任务
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    self._process_batch_results(batch_results)
                    
                    processed_count += len(future_to_batch[future])
                    progress = processed_count / len(image_files) * 100
                    print(f"\r进度: {processed_count}/{len(image_files)} ({progress:.1f}%)", 
                          end='', flush=True)
                    
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"❌ 批次处理失败: {e}")
                    # 标记批次中的所有图片为失败
                    for image_path in batch:
                        self.stats['failed'] += 1
                        error_analysis = {'filename': os.path.basename(image_path), 'error': str(e)}
                        with self.analysis_lock:
                            self.analysis_results.append(error_analysis)
        
        print()  # 换行
    
    def _process_batch_results(self, batch_results: List[Tuple[str, str, Dict]]):
        """处理批次结果"""
        for image_path, category, analysis in batch_results:
            try:
                if category != 'failed':
                    # 移动到对应目录
                    if self.move_image_to_category(image_path, category):
                        self.stats[category] += 1
                        # 如果是符合条件的分类，同时更新总的qualified统计
                        if category.startswith('qualified_'):
                            # 注意：这里不要重复加，因为qualified_*已经包含在qualified统计中
                            pass
                    else:
                        self.stats['failed'] += 1
                        analysis['move_failed'] = True
                else:
                    self.stats['failed'] += 1
                
                # 保存分析结果（线程安全）
                with self.analysis_lock:
                    self.analysis_results.append(analysis)
            
            except Exception as e:
                logger.error(f"❌ 处理结果失败 {image_path}: {e}")
                self.stats['failed'] += 1

def main():
    """主函数"""
    try:
        # 创建配置
        config = OptimizedConfig()
        
        # 检查输入目录
        if not os.path.exists(config.INPUT_DIR):
            logger.error(f"❌ 输入目录不存在: {config.INPUT_DIR}")
            return
        
        # 检查车牌检测模型
        if not os.path.exists(config.PLATE_MODEL_PATH):
            logger.error(f"❌ 车牌检测模型不存在: {config.PLATE_MODEL_PATH}")
            return
        
        # 创建分类器并运行
        classifier = OptimizedFacePlateClassifier(config)
        classifier.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
