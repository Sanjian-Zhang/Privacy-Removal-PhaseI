#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正脸和车牌检测分类器 - 改进版本
使用新的计分系统：
- 一张清晰正脸记2分
- 一张清晰车牌记2分
- 可识别文字10个字段记2分
- 总分>5分认为符合要求

改进点：
1. 修复了代码格式和导入问题
2. 优化了错误处理
3. 改进了模型初始化流程
4. 增强了日志记录
5. 优化了内存管理
6. 确保图片处理过程中不压缩 - 保持原始质量

图片质量保护措施：
- 使用cv2.IMREAD_COLOR确保高质量读取
- 使用shutil.copy2保持文件完整性
- 避免不必要的图像重写操作
- 直接传递文件路径给检测模型
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import gc
from collections import defaultdict

# 配置环境变量 - 必须在导入深度学习库之前设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 启用GPU 0和1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # 减少TensorFlow日志输出

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_plate_classifier.log', encoding='utf-8')
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

# === 参数配置 ===
class Config:
    """配置类"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames68'
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames68'
    PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 新计分系统阈值
    SCORE_THRESHOLD = 5                # 总分阈值（>5分符合要求）
    CLEAR_FACE_SCORE = 2              # 清晰正脸得分
    CLEAR_PLATE_SCORE = 2             # 清晰车牌得分
    TEXT_RECOGNITION_SCORE = 2        # 文字识别得分（有文字即得分）
    # TEXT_FIELDS_THRESHOLD = 10        # 文字字段数量阈值（已删除限制）
    
    # 检测阈值 - 优化后的参数
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
    
    # 图像质量保护设置
    PRESERVE_IMAGE_QUALITY = True      # 确保不压缩图片
    IMAGE_READ_FLAGS = cv2.IMREAD_COLOR  # 使用高质量读取标志
    JPEG_QUALITY = 100                 # JPEG保存质量（如果需要保存）
    PNG_COMPRESSION = 0                # PNG压缩级别（0=无压缩）
    
    # 性能参数
    VERBOSE_LOGGING = True
    GC_FREQUENCY = 100                # 垃圾回收频率
    PROGRESS_UPDATE_FREQUENCY = 50    # 进度更新频率
    
    # 安全性改进
    MAX_PROCESSING_TIME_PER_IMAGE = 30  # 每张图片最大处理时间(秒)
    ENABLE_ERROR_RECOVERY = True        # 启用错误恢复
    BACKUP_ORIGINAL_ON_ERROR = True     # 错误时备份原文件
    
    # 速度优化设置 (不影响图片质量)
    ENABLE_SMART_SKIP = True            # 启用智能跳过
    SKIP_PROCESSED_FILES = True         # 跳过已处理文件
    ENABLE_FAST_PREPROCESSING = True    # 启用快速预处理
    BATCH_DETECTION_SIZE = 4            # 批量检测大小
    ENABLE_RESULT_CACHE = True          # 启用结果缓存
    CACHE_SIZE_LIMIT = 1000             # 缓存大小限制
    MIN_FILE_SIZE = 1024                # 最小文件大小(字节)
    EARLY_STOP_ON_SCORE = True          # 达到高分时提前停止检测
    
    # GPU配置（启用GPU加速）
    USE_GPU = True
    GPU_DEVICE_ID = 0                 # 使用GPU 1（GPU 0正在被占用）
    ENABLE_TORCH_OPTIMIZATION = True
    
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
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis"),
            'backup': os.path.join(cls.OUTPUT_BASE_DIR, "backup")  # 新增：备份目录
        }

class FacePlateClassifier:
    """正脸和车牌检测分类器 - 改进版本"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化分类器"""
        self.config = config or Config()
        self.start_time = time.time()
        
        # 验证图像质量保护设置
        self._verify_image_quality_settings()
        
        # 速度优化：结果缓存
        self.result_cache = {} if self.config.ENABLE_RESULT_CACHE else None
        self.processed_hashes = set()  # 已处理文件的哈希值
        
        # 统计信息
        self.stats = {
            'qualified': 0,           # 符合条件(总分>5)
            'qualified_1_4_faces': 0, # 符合条件且1-4张人脸
            'qualified_5_8_faces': 0, # 符合条件且5-8张人脸
            'qualified_9_plus_faces': 0, # 符合条件且9张人脸以上
            'insufficient_score': 0,  # 分数不够
            'no_content': 0,          # 无任何有效内容
            'failed': 0,              # 处理失败
            'skipped_duplicate': 0,   # 跳过重复文件
            'skipped_small': 0,       # 跳过过小文件
            'skipped_processed': 0,   # 跳过已处理文件
            'cache_hits': 0           # 缓存命中次数
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
        
        # 初始化OCR
        self._initialize_ocr()
        
        # 加载已处理文件信息
        self._load_processed_files()
        
        logger.info("🚀 正脸和车牌检测分类器初始化完成（改进版本）")
        logger.info("📸 图像质量保护：已启用，确保处理过程中不压缩图片")
        logger.info(f"⚡ 速度优化：已启用，包括智能跳过和结果缓存")
    
    def _verify_image_quality_settings(self):
        """验证图像质量保护设置"""
        if self.config.PRESERVE_IMAGE_QUALITY:
            logger.info("✅ 图像质量保护已启用")
            logger.info(f"📋 图像读取标志: cv2.IMREAD_COLOR")
            logger.info(f"📋 JPEG质量设置: {self.config.JPEG_QUALITY}%")
            logger.info(f"📋 PNG压缩级别: {self.config.PNG_COMPRESSION} (0=无压缩)")
        else:
            logger.warning("⚠️  图像质量保护未启用")
    
    def _load_processed_files(self):
        """加载已处理文件信息"""
        try:
            if not self.config.SKIP_PROCESSED_FILES:
                return
            
            # 从各个输出目录收集已处理的文件
            for category_name, category_dir in self.output_dirs.items():
                if category_name == 'analysis' or not os.path.exists(category_dir):
                    continue
                
                for filename in os.listdir(category_dir):
                    if filename.lower().endswith(tuple(self.config.SUPPORTED_FORMATS)):
                        # 移除可能的重命名后缀
                        original_name = filename
                        if '_' in filename:
                            parts = filename.split('_')
                            if len(parts) > 1 and parts[-1].split('.')[0].isdigit():
                                original_name = '_'.join(parts[:-1]) + '.' + parts[-1].split('.')[1]
                        
                        self.processed_hashes.add(original_name)
            
            if self.processed_hashes:
                logger.info(f"📋 已加载 {len(self.processed_hashes)} 个已处理文件的记录")
                
        except Exception as e:
            logger.warning(f"⚠️  加载已处理文件信息失败: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的快速哈希值（用于重复检测）"""
        try:
            import hashlib
            
            # 只读取文件的前1KB和最后1KB来快速计算哈希
            # 这样既快速又能有效检测重复
            hash_md5 = hashlib.md5()
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'rb') as f:
                # 读取前1KB
                chunk = f.read(1024)
                hash_md5.update(chunk)
                
                # 如果文件大于2KB，读取最后1KB
                if file_size > 2048:
                    f.seek(-1024, 2)  # 从文件末尾向前1KB
                    chunk = f.read(1024)
                    hash_md5.update(chunk)
                
                # 添加文件大小到哈希中
                hash_md5.update(str(file_size).encode())
            
            return hash_md5.hexdigest()[:16]  # 只取前16位，足够用于重复检测
            
        except Exception as e:
            logger.debug(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def _should_skip_file(self, image_path: str) -> Tuple[bool, str]:
        """判断是否应该跳过文件处理"""
        try:
            filename = os.path.basename(image_path)
            
            # 检查文件大小
            if self.config.MIN_FILE_SIZE > 0:
                file_size = os.path.getsize(image_path)
                if file_size < self.config.MIN_FILE_SIZE:
                    return True, f"文件过小: {file_size} < {self.config.MIN_FILE_SIZE} bytes"
            
            # 检查是否已处理
            if self.config.SKIP_PROCESSED_FILES and filename in self.processed_hashes:
                return True, "文件已处理"
            
            # 检查重复文件（通过快速哈希）
            if self.config.ENABLE_SMART_SKIP:
                file_hash = self._calculate_file_hash(image_path)
                if file_hash and file_hash in self.processed_hashes:
                    return True, "重复文件（哈希匹配）"
                
                # 记录这个哈希
                if file_hash:
                    self.processed_hashes.add(file_hash)
            
            return False, ""
            
        except Exception as e:
            logger.debug(f"文件跳过检查失败 {image_path}: {e}")
            return False, ""
    
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
                            logger.info("🚀 TensorFlow已配置为GPU模式（内存增长）")
                        except RuntimeError as e:
                            logger.warning(f"⚠️  GPU内存增长配置失败: {e}")
                else:
                    tf.config.set_visible_devices([], 'GPU')
                    logger.info("🔧 TensorFlow已配置为CPU模式")
            except ImportError:
                logger.info("ℹ️  未检测到TensorFlow")
            
            # 初始化RetinaFace
            if 'cuda' in self.device:
                logger.info("🔍 初始化RetinaFace模型（GPU模式）...")
            else:
                logger.info("🔍 初始化RetinaFace模型（CPU模式）...")
            self._test_retinaface()
            
            # 初始化车牌检测模型
            logger.info("🚗 初始化车牌检测模型...")
            self._initialize_plate_model()
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def _test_retinaface(self):
        """测试RetinaFace模型"""
        try:
            # 创建测试图像
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            
            # 尝试检测
            result = RetinaFace.detect_faces(test_img)
            logger.info("✅ RetinaFace模型初始化成功（CPU模式）")
            
        except Exception as e:
            logger.error(f"❌ RetinaFace初始化失败: {e}")
            logger.info("🔄 尝试重新初始化RetinaFace...")
            
            # 清理状态
            try:
                # 尝试清理TensorFlow会话，但忽略任何错误
                pass
            except:
                pass
            
            # 重新尝试
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            RetinaFace.detect_faces(test_img)
            logger.info("✅ RetinaFace模型重新初始化成功")
    
    def _initialize_plate_model(self):
        """初始化车牌检测模型"""
        if not os.path.exists(self.config.PLATE_MODEL_PATH):
            raise FileNotFoundError(f"车牌检测模型文件不存在: {self.config.PLATE_MODEL_PATH}")
        
        # 初始化YOLO模型（使用配置的设备）
        self.plate_model = YOLO(self.config.PLATE_MODEL_PATH)
        
        if 'cuda' in self.device:
            logger.info(f"✅ 车牌检测模型初始化成功（GPU模式 - {self.device}）")
        else:
            logger.info("✅ 车牌检测模型初始化成功（CPU模式）")
    
    def _initialize_ocr(self):
        """初始化OCR模型"""
        try:
            logger.info("📝 初始化EasyOCR模型...")
            
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
    
    def calculate_face_clarity(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """计算人脸区域的清晰度"""
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
            
            # 使用拉普拉斯算子计算清晰度
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # 额外的清晰度指标：梯度幅值
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # 综合清晰度分数
            clarity_score = laplacian_var + gradient_mean * 0.1
            
            return float(clarity_score)
            
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
    
    def detect_faces(self, image_path: str) -> Tuple[int, List[Dict]]:
        """使用RetinaFace检测清晰正脸，确保不压缩原图"""
        try:
            # 直接传递图片路径给RetinaFace，避免重复读取和潜在的压缩
            detections = RetinaFace.detect_faces(image_path)
            
            if not isinstance(detections, dict) or len(detections) == 0:
                return 0, []
            
            # 只在需要时读取原图进行质量评估
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 明确指定读取彩色图像
            if img is None:
                return 0, []
            
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
            
            return len(clear_frontal_faces), clear_frontal_faces
            
        except Exception as e:
            logger.debug(f"人脸检测失败 {image_path}: {e}")
            return 0, []
    
    def detect_license_plates(self, image_path: str) -> Tuple[int, List[Dict]]:
        """使用YOLO检测清晰车牌"""
        try:
            # 使用配置的设备进行推理
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
            
            return len(clear_plates), clear_plates
            
        except Exception as e:
            logger.debug(f"车牌检测失败 {image_path}: {e}")
            return 0, []
    
    def detect_text(self, image_path: str) -> Tuple[int, List[Dict]]:
        """使用OCR检测可识别的文字，确保不压缩原图"""
        try:
            if self.ocr_reader is None:
                return 0, []
            
            # 直接读取原图，使用最高质量设置
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 明确指定读取彩色图像
            if img is None:
                return 0, []
            
            results = self.ocr_reader.readtext(img)
            
            if not results:
                return 0, []
            
            valid_texts = []
            
            for bbox, text, confidence in results:
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
            
            return len(valid_texts), valid_texts
            
        except Exception as e:
            logger.debug(f"文字检测失败 {image_path}: {e}")
            return 0, []
    
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
    
    def classify_image(self, image_path: str) -> Tuple[str, Dict]:
        """使用新计分系统分类图像，包含速度优化"""
        start_time = time.time()
        
        try:
            filename = os.path.basename(image_path)
            
            # 速度优化：检查是否应该跳过
            should_skip, skip_reason = self._should_skip_file(image_path)
            if should_skip:
                if "文件已处理" in skip_reason:
                    self.stats['skipped_processed'] += 1
                elif "文件过小" in skip_reason:
                    self.stats['skipped_small'] += 1
                elif "重复文件" in skip_reason:
                    self.stats['skipped_duplicate'] += 1
                
                return 'skipped', {
                    'filename': filename, 
                    'skip_reason': skip_reason,
                    'processing_time': time.time() - start_time
                }
            
            # 检查缓存
            if self.result_cache:
                cache_key = self._get_cache_key(image_path)
                if cache_key in self.result_cache:
                    self.stats['cache_hits'] += 1
                    cached_result = self.result_cache[cache_key].copy()
                    cached_result['from_cache'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    return cached_result['category'], cached_result
            
            # 检查文件是否存在和有效
            if not os.path.exists(image_path):
                logger.error(f"❌ 文件不存在: {image_path}")
                return 'failed', {'filename': filename, 'error': '文件不存在'}
            
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                logger.error(f"❌ 文件为空: {image_path}")
                return 'failed', {'filename': filename, 'error': '文件为空'}
            
            # 智能检测顺序：先检测容易得分的项目
            score = 0
            score_details = []
            frontal_count = 0
            plate_count = 0 
            text_count = 0
            face_details = []
            plate_details = []
            text_details = []
            
            # 1. 首先检测车牌（通常更快且容易检测）
            plate_count, plate_details = self.detect_license_plates(image_path)
            if plate_count > 0:
                plate_score = plate_count * self.config.CLEAR_PLATE_SCORE
                score += plate_score
                score_details.append(f"清晰车牌 {plate_count} 张 × {self.config.CLEAR_PLATE_SCORE} = {plate_score} 分")
            
            # 检查处理时间
            if time.time() - start_time > self.config.MAX_PROCESSING_TIME_PER_IMAGE:
                logger.warning(f"⚠️  图像处理超时: {image_path}")
                return 'failed', {'filename': filename, 'error': '处理超时'}
            
            # 2. 检测文字（OCR相对较快）
            text_count, text_details = self.detect_text(image_path)
            if text_count > 0:
                text_score = self.config.TEXT_RECOGNITION_SCORE
                score += text_score
                score_details.append(f"可识别文字 {text_count} 个字段 = {text_score} 分")
            
            # 早期停止：如果已经达到高分，可以跳过人脸检测
            if self.config.EARLY_STOP_ON_SCORE and score > self.config.SCORE_THRESHOLD + 2:
                logger.debug(f"早期停止：分数已足够高 ({score})")
            else:
                # 3. 最后检测人脸（通常最耗时）
                if time.time() - start_time < self.config.MAX_PROCESSING_TIME_PER_IMAGE:
                    frontal_count, face_details = self.detect_faces(image_path)
                    if frontal_count > 0:
                        face_score = frontal_count * self.config.CLEAR_FACE_SCORE
                        score += face_score
                        score_details.append(f"清晰正脸 {frontal_count} 张 × {self.config.CLEAR_FACE_SCORE} = {face_score} 分")
            
            # 判断是否符合要求
            meets_requirements = score > self.config.SCORE_THRESHOLD
            processing_time = time.time() - start_time
            
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
                'processing_time': processing_time,
                'file_size': file_size,
                'timestamp': time.time()
            }
            
            # 分类逻辑
            if meets_requirements:
                face_category = self.get_face_count_category(frontal_count)
                category = face_category
                analysis['qualification_reason'] = f'总分 {score} 分 > {self.config.SCORE_THRESHOLD} 分，符合要求'
                analysis['face_count_category'] = face_category
                
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
            
            # 缓存结果
            if self.result_cache and len(self.result_cache) < self.config.CACHE_SIZE_LIMIT:
                cache_key = self._get_cache_key(image_path)
                self.result_cache[cache_key] = analysis.copy()
            
            return category, analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像分类失败 {image_path}: {e}")
            return 'failed', {
                'filename': os.path.basename(image_path), 
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _get_cache_key(self, image_path: str) -> str:
        """生成缓存键"""
        try:
            stat = os.stat(image_path)
            return f"{os.path.basename(image_path)}_{stat.st_size}_{int(stat.st_mtime)}"
        except:
            return os.path.basename(image_path)
    
    def move_image_to_category(self, image_path: str, category: str) -> bool:
        """移动图像到对应分类目录，确保不压缩图片，包含错误恢复机制"""
        backup_path = None  # 初始化备份路径
        
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
            
            # 安全移动：先备份（如果启用），再复制，最后删除原文件
            if self.config.BACKUP_ORIGINAL_ON_ERROR:
                backup_dir = self.output_dirs.get('backup')
                if backup_dir and os.path.exists(backup_dir):
                    backup_path = os.path.join(backup_dir, f"backup_{int(time.time())}_{filename}")
                    shutil.copy2(image_path, backup_path)
            
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
            
            # 如果有备份且操作成功，删除备份
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except:
                    pass  # 备份删除失败不影响主流程
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动图像失败 {image_path}: {e}")
            
            # 错误恢复：如果有备份，尝试恢复
            if self.config.ENABLE_ERROR_RECOVERY and backup_path and os.path.exists(backup_path):
                try:
                    if not os.path.exists(image_path):
                        shutil.move(backup_path, image_path)
                        logger.info(f"✅ 已从备份恢复文件: {image_path}")
                except Exception as recovery_error:
                    logger.error(f"❌ 错误恢复失败: {recovery_error}")
            
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
            analysis_file = os.path.join(analysis_dir, "classification_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            # 保存统计摘要
            summary = {
                'total_processed': len(self.analysis_results),
                'statistics': self.stats.copy(),
                'processing_time': time.time() - self.start_time,
                'scoring_system': {
                    'clear_face_score': self.config.CLEAR_FACE_SCORE,
                    'clear_plate_score': self.config.CLEAR_PLATE_SCORE,
                    'text_recognition_score': self.config.TEXT_RECOGNITION_SCORE,
                    'score_threshold': self.config.SCORE_THRESHOLD
                },
                'configuration': {
                    'yaw_angle_threshold': self.config.YAW_ANGLE_THRESHOLD,
                    'min_face_confidence': self.config.MIN_FACE_CONFIDENCE,
                    'min_plate_confidence': self.config.MIN_PLATE_CONFIDENCE,
                    'min_text_confidence': self.config.MIN_TEXT_CONFIDENCE
                },
                'timestamp': time.time()
            }
            
            summary_file = os.path.join(analysis_dir, "classification_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 分析结果已保存到: {analysis_file}")
            logger.info(f"📊 统计摘要已保存到: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存分析结果失败: {e}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        processing_time = time.time() - self.start_time
        # 修正总数计算：不重复计算qualified的子分类，并包含跳过的文件
        total_processed = (self.stats['qualified'] + self.stats['insufficient_score'] + 
                          self.stats['no_content'] + self.stats['failed'] +
                          self.stats['skipped_duplicate'] + self.stats['skipped_small'] + 
                          self.stats['skipped_processed'])
        
        logger.info("="*80)
        logger.info("🎉 正脸和车牌分类完成！最终统计:")
        logger.info("新的计分系统:")
        logger.info(f"  - 一张清晰正脸 = {self.config.CLEAR_FACE_SCORE} 分")
        logger.info(f"  - 一张清晰车牌 = {self.config.CLEAR_PLATE_SCORE} 分")  
        logger.info(f"  - 可识别文字 = {self.config.TEXT_RECOGNITION_SCORE} 分")
        logger.info(f"  - 总分 > {self.config.SCORE_THRESHOLD} 分 = 符合要求")
        logger.info(f"✅ 符合条件总计(>{self.config.SCORE_THRESHOLD}分): {self.stats['qualified']:,}")
        logger.info(f"  📸 1-4张人脸: {self.stats['qualified_1_4_faces']:,}")
        logger.info(f"  👥 5-8张人脸: {self.stats['qualified_5_8_faces']:,}")
        logger.info(f"  👨‍👩‍👧‍👦 9张人脸以上: {self.stats['qualified_9_plus_faces']:,}")
        logger.info(f"❌ 分数不够(≤{self.config.SCORE_THRESHOLD}分): {self.stats['insufficient_score']:,}")
        logger.info(f"❌ 无任何内容: {self.stats['no_content']:,}")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,}")
        
        # 速度优化统计
        logger.info(f"⚡ 速度优化统计:")
        logger.info(f"  � 跳过重复文件: {self.stats['skipped_duplicate']:,}")
        logger.info(f"  📏 跳过过小文件: {self.stats['skipped_small']:,}")
        logger.info(f"  ✅ 跳过已处理文件: {self.stats['skipped_processed']:,}")
        logger.info(f"  💾 缓存命中: {self.stats['cache_hits']:,}")
        
        total_skipped = (self.stats['skipped_duplicate'] + self.stats['skipped_small'] + 
                        self.stats['skipped_processed'])
        if total_skipped > 0:
            logger.info(f"  �📊 总跳过文件: {total_skipped:,}")
            logger.info(f"  ⚡ 跳过率: {(total_skipped/total_processed)*100:.1f}%")
        
        logger.info(f"📊 总处理数量: {total_processed:,}")
        logger.info(f"⏰ 总耗时: {processing_time:.1f}秒")
        
        if total_processed > 0:
            # 计算实际处理的文件数（排除跳过的）
            actually_processed = total_processed - total_skipped
            if actually_processed > 0:
                avg_speed = actually_processed / processing_time
                logger.info(f"🚀 实际处理速度: {avg_speed:.1f} 张/秒")
            
            # 包含跳过文件的总体速度
            total_speed = total_processed / processing_time
            logger.info(f"🚀 总体处理速度: {total_speed:.1f} 张/秒")
            
            success_rate = (self.stats['qualified'] / actually_processed) * 100 if actually_processed > 0 else 0
            logger.info(f"📈 符合条件比例: {success_rate:.1f}%")
        
        # 显示各目录文件数量
        logger.info("\n📂 各分类目录统计:")
        categories = [
            ("符合条件-1-4张人脸", self.output_dirs['qualified_1_4_faces']),
            ("符合条件-5-8张人脸", self.output_dirs['qualified_5_8_faces']),
            ("符合条件-9张人脸以上", self.output_dirs['qualified_9_plus_faces']),
            ("分数不够", self.output_dirs['insufficient_score']),
            ("无任何内容", self.output_dirs['no_content'])
        ]
        
        for name, dir_path in categories:
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) 
                           if f.lower().endswith(tuple(self.config.SUPPORTED_FORMATS))])
                logger.info(f"  📁 {name}: {count} 张图片")
        
        logger.info("="*80)
    
    def run(self):
        """运行分类器"""
        logger.info("🚀 启动正脸和车牌检测分类器（改进版本）...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📁 输出目录: {self.config.OUTPUT_BASE_DIR}")
        logger.info(f"💻 计算设备: {self.device}")
        logger.info(f"📊 计分规则:")
        logger.info(f"  - 清晰正脸: {self.config.CLEAR_FACE_SCORE} 分/张")
        logger.info(f"  - 清晰车牌: {self.config.CLEAR_PLATE_SCORE} 分/张")
        logger.info(f"  - 可识别文字: {self.config.TEXT_RECOGNITION_SCORE} 分(有文字即得分)")
        logger.info(f"  - 通过阈值: > {self.config.SCORE_THRESHOLD} 分")
        logger.info(f"📐 yaw角度阈值: {self.config.YAW_ANGLE_THRESHOLD}°")
        logger.info(f"🎯 人脸置信度阈值: {self.config.MIN_FACE_CONFIDENCE}")
        logger.info(f"🎯 车牌置信度阈值: {self.config.MIN_PLATE_CONFIDENCE}")
        logger.info(f"🎯 文字识别置信度阈值: {self.config.MIN_TEXT_CONFIDENCE}")
        logger.info(f"🔍 最小清晰度分数: {self.config.MIN_FACE_CLARITY_SCORE}")
        logger.info(f"📏 最小人脸面积: {self.config.FACE_AREA_THRESHOLD}px²")
        logger.info(f"⏱️  单图片最大处理时间: {self.config.MAX_PROCESSING_TIME_PER_IMAGE}秒")
        logger.info(f"🛡️  错误恢复: {'启用' if self.config.ENABLE_ERROR_RECOVERY else '禁用'}")
        logger.info(f"💾 原文件备份: {'启用' if self.config.BACKUP_ORIGINAL_ON_ERROR else '禁用'}")
        
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
            timeout_files = []
            
            # 改进的进度显示
            for i, image_path in enumerate(image_files):
                current_time = time.time()
                
                try:
                    # 分类图像
                    category, analysis = self.classify_image(image_path)
                    
                    # 处理跳过的文件
                    if category == 'skipped':
                        # 跳过的文件不需要移动，统计已在classify_image中更新
                        pass
                    elif category != 'failed':
                        # 移动到对应目录
                        if self.move_image_to_category(image_path, category):
                            self.stats[category] += 1
                            # 如果是符合条件的分类，同时更新总的qualified统计
                            if category.startswith('qualified_'):
                                self.stats['qualified'] += 1
                        else:
                            self.stats['failed'] += 1
                            analysis['move_failed'] = True
                            failed_files.append(image_path)
                    else:
                        self.stats['failed'] += 1
                        failed_files.append(image_path)
                    
                    # 记录超时文件
                    if analysis.get('error') == '处理超时':
                        timeout_files.append(image_path)
                    
                    # 保存分析结果（跳过的文件也记录）
                    self.analysis_results.append(analysis)
                    
                    # 智能进度显示
                    if (i + 1) % self.config.PROGRESS_UPDATE_FREQUENCY == 0 or i == total_files - 1:
                        progress = (i + 1) / total_files * 100
                        elapsed = current_time - start_time
                        
                        if i > 0:
                            avg_time = elapsed / (i + 1)
                            remaining_files = total_files - (i + 1)
                            eta = remaining_files * avg_time
                            
                            # 格式化ETA
                            if eta > 3600:
                                eta_str = f"{eta/3600:.1f}小时"
                            elif eta > 60:
                                eta_str = f"{eta/60:.1f}分钟"
                            else:
                                eta_str = f"{eta:.0f}秒"
                            
                            print(f"\r📊 进度: {i+1}/{total_files} ({progress:.1f}%) "
                                  f"| 已用时: {elapsed/60:.1f}分钟 "
                                  f"| 预计剩余: {eta_str} "
                                  f"| 速度: {(i+1)/elapsed:.1f}张/秒", end='', flush=True)
                        else:
                            print(f"\r📊 进度: {i+1}/{total_files} ({progress:.1f}%)", end='', flush=True)
                    
                    # 定期内存清理
                    if (i + 1) % self.config.GC_FREQUENCY == 0:
                        gc.collect()
                        # 如果使用GPU，清理GPU内存
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
            
            # 显示处理摘要
            total_time = time.time() - start_time
            if failed_files:
                logger.warning(f"⚠️  失败文件数: {len(failed_files)}")
                if len(failed_files) <= 10:
                    for failed_file in failed_files:
                        logger.warning(f"  - {os.path.basename(failed_file)}")
                else:
                    logger.warning(f"  显示前10个: {[os.path.basename(f) for f in failed_files[:10]]}")
            
            if timeout_files:
                logger.warning(f"⏱️  超时文件数: {len(timeout_files)}")
        
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
        
        # 检查车牌检测模型
        if not os.path.exists(config.PLATE_MODEL_PATH):
            logger.error(f"❌ 车牌检测模型不存在: {config.PLATE_MODEL_PATH}")
            return
        
        # 创建分类器并运行
        classifier = FacePlateClassifier(config)
        classifier.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
