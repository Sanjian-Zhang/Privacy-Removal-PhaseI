#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级人脸车牌检测器 - 优化版本
增加了以下优化策略：
1. 批量处理 - 一次处理多张图片
2. 单进程模式 - 避免CUDA多进程初始化冲突
3. GPU批量推理 - 减少GPU上下文切换
4. 预过滤机制 - 快速跳过明显不符合条件的图片
5. 内存池管理 - 减少内存分配开销
6. 智能分组 - 按图片大小/类型分组处理

修复说明：
- 使用spawn启动方法避免CUDA多进程问题
- 改用单进程批处理模式提高稳定性
- 增强GPU初始化错误处理和回退机制
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import hashlib
import sys

# 设置环境变量 - 必须在导入深度学习库之前
os.environ['CUDA_VISIBLE_DEVICES'] = '0'     # 只使用GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # 减少TensorFlow日志输出

# 修复CUDA多进程问题 - 设置spawn启动方法
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_face_plate_detector_optimized.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ProgressBar:
    """简单的进度条显示器"""
    
    def __init__(self, total: int, prefix: str = "Progress", length: int = 50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: Optional[int] = None):
        """更新进度条"""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        # 避免过于频繁的更新
        now = time.time()
        if now - self.last_update < 0.1 and self.current < self.total:  # 每100ms更新一次
            return
        
        self.last_update = now
        
        # 计算进度
        progress = self.current / self.total if self.total > 0 else 0
        percent = progress * 100
        
        # 计算时间信息
        elapsed = now - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            if speed > 0:
                eta = (self.total - self.current) / speed
                eta_str = self._format_time(eta)
            else:
                eta_str = "N/A"
        else:
            speed = 0
            eta_str = "N/A"
        
        # 绘制进度条
        filled_length = int(self.length * progress)
        bar = '█' * filled_length + '░' * (self.length - filled_length)
        
        # 格式化输出
        elapsed_str = self._format_time(elapsed)
        speed_str = f"{speed:.1f}/s" if speed > 0 else "0/s"
        
        # 输出进度条
        print(f'\r{self.prefix}: |{bar}| {self.current}/{self.total} '
              f'({percent:.1f}%) 用时:{elapsed_str} 速度:{speed_str} 剩余:{eta_str}', 
              end='', flush=True)
        
        # 完成时换行
        if self.current >= self.total:
            print()
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m{secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def finish(self):
        """完成进度条"""
        self.update(self.total)

class ProgressTracker:
    """多阶段进度跟踪器"""
    
    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.overall_start = time.time()
    
    def add_stage(self, name: str, total: int, prefix: Optional[str] = None):
        """添加一个处理阶段"""
        if prefix is None:
            prefix = name
        self.stages[name] = {
            'progress_bar': ProgressBar(total, prefix),
            'total': total,
            'completed': False
        }
    
    def start_stage(self, name: str):
        """开始一个阶段"""
        if name in self.stages:
            self.current_stage = name
            print(f"\n🔄 开始阶段: {name}")
        else:
            print(f"⚠️  未知阶段: {name}")
    
    def update_stage(self, name: Optional[str] = None, current: Optional[int] = None):
        """更新当前阶段进度"""
        stage_name = name or self.current_stage
        if stage_name and stage_name in self.stages:
            self.stages[stage_name]['progress_bar'].update(current)
    
    def finish_stage(self, name: Optional[str] = None):
        """完成一个阶段"""
        stage_name = name or self.current_stage
        if stage_name and stage_name in self.stages:
            self.stages[stage_name]['progress_bar'].finish()
            self.stages[stage_name]['completed'] = True
            elapsed = time.time() - self.overall_start
            print(f"✅ 阶段 '{stage_name}' 完成 (总用时: {self._format_time(elapsed)})")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m{secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def print_summary(self):
        """打印所有阶段的摘要"""
        total_time = time.time() - self.overall_start
        print(f"\n📊 处理摘要 (总时间: {self._format_time(total_time)}):")
        for name, stage in self.stages.items():
            status = "✅ 完成" if stage['completed'] else "❌ 未完成"
            print(f"  {name}: {status}")

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

class OptimizedConfig:
    """优化配置类"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames87'  # 输入目录
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames87'  # 输出基础目录
    
    # 模型路径
    YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
    LICENSE_PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 检测阈值
    REQUIRED_CLOSE_FRONTAL_FACES = 4      # 需要的近景正脸数量（保留用于向下兼容）
    MIN_FACE_CONFIDENCE_YOLO = 0.5        # YOLO人脸最小置信度
    MIN_FACE_CONFIDENCE_RETINA = 0.8      # RetinaFace最小置信度
    MIN_PLATE_CONFIDENCE = 0.5            # 车牌最小置信度
    MIN_TEXT_CONFIDENCE = 0.5             # 文字最小置信度
    YAW_ANGLE_THRESHOLD = 30.0            # yaw角度阈值（正脸）
    
    # 新的评分系统
    SCORE_PER_CLEAR_FRONTAL_FACE = 2      # 每张清晰正脸的分数
    SCORE_PER_CLEAR_PLATE = 2             # 每张清晰车牌的分数
    SCORE_PER_TEXT = 1                    # 可识别文字的分数
    REQUIRED_TOTAL_SCORE = 5              # 需要的最低总分数
    
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
    PREFERRED_GPU = 0                     # 只使用GPU 0
    FALLBACK_GPU = 0                      # 备用也是GPU 0
    
    # 文件格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 优化参数
    BATCH_SIZE = 32                       # 进一步增大批处理大小提高效率
    MAX_WORKERS = 1                       # 使用单进程避免CUDA多进程问题
    THREAD_POOL_SIZE = 2                  # 减少线程池大小
    PREFILTER_ENABLED = False             # 暂时禁用预过滤以加快速度
    SMART_GROUPING = False                # 暂时禁用智能分组以简化处理
    USE_MULTIPROCESSING = False           # 禁用多进程，使用单进程批处理
    PREFILTER_SAMPLE_RATE = 0.1           # 如果启用预过滤，只对10%的图片进行预过滤
    
    # 分组参数
    SMALL_IMAGE_THRESHOLD = 500 * 500     # 小图片阈值
    MEDIUM_IMAGE_THRESHOLD = 1000 * 1000  # 中等图片阈值
    LARGE_IMAGE_THRESHOLD = 2000 * 2000   # 大图片阈值
    
    # 性能参数
    GC_FREQUENCY = 100                    # 垃圾回收频率
    PROGRESS_UPDATE_FREQUENCY = 50        # 进度更新频率
    MAX_PROCESSING_TIME_PER_IMAGE = 60    # 单张图片最大处理时间
    MEMORY_LIMIT_GB = 8                   # 内存限制（GB）
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'high_score': os.path.join(cls.OUTPUT_BASE_DIR, "high_score_images"),  # 总分>5分的图片
            'low_score': os.path.join(cls.OUTPUT_BASE_DIR, "low_score_images"),   # 1-5分的图片
            'zero_score': os.path.join(cls.OUTPUT_BASE_DIR, "zero_score_images"), # 0分的图片
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis"),
            # 保留原分类目录以备兼容
            'satisfied_4_faces': os.path.join(cls.OUTPUT_BASE_DIR, "satisfied_4_close_frontal_faces"),
            'satisfied_with_plate': os.path.join(cls.OUTPUT_BASE_DIR, "satisfied_with_plate_text"),
            'insufficient_faces': os.path.join(cls.OUTPUT_BASE_DIR, "insufficient_faces"),
            'no_faces': os.path.join(cls.OUTPUT_BASE_DIR, "no_faces"),
        }

class ImageGrouper:
    """图片分组器 - 按大小和特征智能分组"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
    
    def get_image_info(self, image_path: str) -> Dict:
        """获取图片基本信息"""
        try:
            # 获取文件信息
            stat = os.stat(image_path)
            file_size = stat.st_size
            
            # 快速读取图片尺寸（不加载完整图片）
            with open(image_path, 'rb') as f:
                # 尝试快速解析图片尺寸
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    height, width = img.shape
                    pixel_count = width * height
                else:
                    # 备用方案：使用PIL获取尺寸
                    from PIL import Image
                    with Image.open(image_path) as pil_img:
                        width, height = pil_img.size
                        pixel_count = width * height
            
            return {
                'path': image_path,
                'file_size': file_size,
                'width': width,
                'height': height,
                'pixel_count': pixel_count,
                'aspect_ratio': width / height if height > 0 else 1.0
            }
        except Exception as e:
            logger.debug(f"获取图片信息失败 {image_path}: {e}")
            return {
                'path': image_path,
                'file_size': 0,
                'width': 0,
                'height': 0,
                'pixel_count': 0,
                'aspect_ratio': 1.0
            }
    
    def group_by_size(self, image_files: List[str]) -> Dict[str, List[str]]:
        """按图片大小分组"""
        groups = {
            'small': [],      # 小图片
            'medium': [],     # 中等图片
            'large': [],      # 大图片
            'extra_large': [] # 超大图片
        }
        
        logger.info("📊 按图片大小分组...")
        
        for image_path in image_files:
            info = self.get_image_info(image_path)
            pixel_count = info['pixel_count']
            
            if pixel_count <= self.config.SMALL_IMAGE_THRESHOLD:
                groups['small'].append(image_path)
            elif pixel_count <= self.config.MEDIUM_IMAGE_THRESHOLD:
                groups['medium'].append(image_path)
            elif pixel_count <= self.config.LARGE_IMAGE_THRESHOLD:
                groups['large'].append(image_path)
            else:
                groups['extra_large'].append(image_path)
        
        # 显示分组统计
        for group_name, files in groups.items():
            if files:
                logger.info(f"  📸 {group_name}: {len(files)} 张图片")
        
        return groups
    
    def group_by_batch(self, image_files: List[str], batch_size: int) -> List[List[str]]:
        """按批次分组"""
        batches = []
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"📦 创建 {len(batches)} 个批次，每批最多 {batch_size} 张图片")
        return batches
    
    def create_balanced_groups(self, image_files: List[str], num_workers: int) -> List[List[str]]:
        """创建负载均衡的分组"""
        if not image_files:
            return []
        
        # 获取所有图片信息
        image_infos = [self.get_image_info(path) for path in image_files]
        
        # 按处理复杂度排序（大图片处理时间更长）
        image_infos.sort(key=lambda x: x['pixel_count'], reverse=True)
        
        # 创建均衡分组
        groups = [[] for _ in range(num_workers)]
        group_loads = [0] * num_workers
        
        for info in image_infos:
            # 找到负载最小的组
            min_load_idx = min(range(num_workers), key=lambda i: group_loads[i])
            groups[min_load_idx].append(info['path'])
            group_loads[min_load_idx] += info['pixel_count']
        
        # 过滤空组
        groups = [group for group in groups if group]
        
        logger.info(f"⚖️  创建 {len(groups)} 个均衡分组")
        for i, group in enumerate(groups):
            if group:
                avg_pixels = group_loads[i] / len(group) if group else 0
                logger.info(f"  📸 组 {i+1}: {len(group)} 张图片，平均像素: {avg_pixels:,.0f}")
        
        return groups

class PreFilter:
    """预过滤器 - 快速过滤明显不符合条件的图片"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
    
    def quick_face_check(self, image_path: str) -> Tuple[bool, str]:
        """快速人脸检查 - 使用简单方法预判是否可能有人脸"""
        try:
            # 首先检查文件大小（太小的文件可能没有清晰内容）
            file_size = os.path.getsize(image_path)
            if file_size < 50000:  # 小于50KB
                return False, "文件太小"
            
            # 快速读取图片基本信息
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "无法读取图片"
            
            # 检查图片尺寸
            height, width = img.shape
            if min(height, width) < 300:
                return False, "分辨率太低"
            
            # 快速边缘检测（替代Haar级联）
            # 缩小图片进行快速分析
            scale_factor = 0.2  # 更大的缩放比例
            small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
            
            # 计算边缘密度和对比度
            edges = cv2.Canny(small_img, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 计算图像对比度
            contrast = np.std(small_img)
            
            # 简单的启发式规则
            if edge_density < 0.05:  # 边缘太少
                return False, "边缘密度不足"
            
            if contrast < 20:  # 对比度太低
                return False, "对比度不足"
            
            # 检查图像是否过于模糊
            laplacian_var = cv2.Laplacian(small_img, cv2.CV_64F).var()
            if laplacian_var < 50:
                return False, "图像模糊"
            
            return True, "通过快速检查"
        
        except Exception as e:
            logger.debug(f"预过滤失败 {image_path}: {e}")
            return True, "预过滤异常，保留"

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = None
        self.models = {}
        self.ocr_reader = None
        
    def initialize_models(self, device: str):
        """初始化模型（在子进程中调用）"""
        try:
            # 先检查CUDA可用性
            if torch.cuda.is_available():
                # 设置GPU设备
                torch.cuda.set_device(0)  # 强制使用GPU 0
                self.device = 'cuda:0'
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 设置GPU内存管理
                torch.cuda.set_per_process_memory_fraction(0.7, device=0)  # 使用70%的GPU内存
                
                logger.info(f"🔧 GPU初始化成功，使用设备: {self.device}")
            else:
                self.device = 'cpu'
                logger.warning("⚠️  CUDA不可用，使用CPU模式")
            
            # 初始化YOLO模型
            logger.info("🔄 初始化YOLO人脸检测模型...")
            self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
            self.models['face'].to(self.device)
            
            logger.info("🔄 初始化YOLO车牌检测模型...")
            self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
            self.models['plate'].to(self.device)
            
            # 初始化OCR
            logger.info("🔄 初始化OCR模型...")
            gpu_enabled = 'cuda' in self.device
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=gpu_enabled)
            
            device_info = f"{self.device} ({'GPU' if 'cuda' in self.device else 'CPU'}模式)"
            logger.info(f"✅ 批处理器模型初始化完成 ({device_info})")
            return True
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"❌ CUDA初始化失败: {e}")
                logger.info("🔄 尝试使用CPU模式...")
                try:
                    self.device = 'cpu'
                    self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
                    self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
                    self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    logger.info(f"✅ 批处理器模型初始化完成 (CPU模式)")
                    return True
                except Exception as e2:
                    logger.error(f"❌ CPU模式初始化也失败: {e2}")
                    return False
            else:
                raise e
        except Exception as e:
            logger.error(f"❌ 批处理器模型初始化失败: {e}")
            # 如果GPU初始化失败，尝试使用CPU
            try:
                logger.info("🔄 尝试使用CPU模式...")
                self.device = 'cpu'
                self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
                self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                logger.info(f"✅ 批处理器模型初始化完成 (CPU模式)")
                return True
            except Exception as e2:
                logger.error(f"❌ CPU模式初始化也失败: {e2}")
                return False
    
    def process_batch(self, image_paths: List[str]) -> List[Tuple[str, Dict]]:
        """批量处理图片"""
        results = []
        
        try:
            # 批量YOLO检测
            face_results = self.models['face'](image_paths, verbose=False, device=self.device)
            
            for i, image_path in enumerate(image_paths):
                try:
                    # 处理单张图片
                    result = self.process_single_image(
                        image_path, 
                        face_results[i] if i < len(face_results) else None
                    )
                    results.append((image_path, result))
                    
                except Exception as e:
                    logger.error(f"批处理中单张图片失败 {image_path}: {e}")
                    results.append((image_path, {'error': str(e)}))
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            # 如果批处理失败，尝试单独处理每张图片
            for image_path in image_paths:
                try:
                    result = self.process_single_image(image_path, None)
                    results.append((image_path, result))
                except Exception as e2:
                    results.append((image_path, {'error': str(e2)}))
        
        return results
    
    def process_single_image(self, image_path: str, yolo_result=None) -> Dict:
        """处理单张图片 - 使用新的评分系统"""
        try:
            filename = os.path.basename(image_path)
            start_time = time.time()
            
            # 初始化分数和检测结果
            total_score = 0
            frontal_face_count = 0
            clear_plate_count = 0
            text_count = 0
            
            # 详细检测结果
            detection_details = {
                'yolo_faces': [],
                'retina_faces': [],
                'plates': [],
                'texts': []
            }
            
            # 1. 使用RetinaFace检测清晰正脸
            try:
                detections = RetinaFace.detect_faces(image_path)
                
                if isinstance(detections, dict) and len(detections) > 0:
                    # 读取图像信息
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        img_area = img_width * img_height
                        
                        for face_key, face_data in detections.items():
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
                            
                            # 计算yaw角度判断是否为正脸
                            yaw_angle = self.calculate_yaw_angle(landmarks)
                            is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                            
                            # 判断是否为近景
                            area_ratio = face_area / img_area
                            is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                            
                            # 只有清晰的近景正脸才计分
                            if is_frontal and is_close_up:
                                frontal_face_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_FRONTAL_FACE
                            
                            face_info = {
                                'confidence': confidence,
                                'yaw_angle': yaw_angle,
                                'is_frontal': is_frontal,
                                'area_ratio': area_ratio,
                                'is_close_up': is_close_up,
                                'scored': is_frontal and is_close_up
                            }
                            detection_details['retina_faces'].append(face_info)
                            
            except Exception as e:
                logger.debug(f"RetinaFace检测失败 {image_path}: {e}")
            
            # 2. 检测清晰车牌
            try:
                plate_results = self.models['plate']([image_path], verbose=False, device=self.device)
                
                if plate_results and len(plate_results) > 0:
                    result = plate_results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            if confidence >= self.config.MIN_PLATE_CONFIDENCE:
                                clear_plate_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_PLATE
                                
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                plate_info = {
                                    'confidence': confidence,
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'scored': True
                                }
                                detection_details['plates'].append(plate_info)
                                
            except Exception as e:
                logger.debug(f"车牌检测失败 {image_path}: {e}")
            
            # 3. 检测可识别文字
            try:
                if self.ocr_reader is not None:
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        ocr_results = self.ocr_reader.readtext(img)
                        
                        if ocr_results:
                            for bbox, text, confidence in ocr_results:
                                confidence = float(confidence) if confidence is not None else 0.0
                                
                                if confidence >= self.config.MIN_TEXT_CONFIDENCE:
                                    cleaned_text = text.strip()
                                    if len(cleaned_text) >= 2:  # 至少2个字符才算有效文字
                                        text_count += 1
                                        total_score += self.config.SCORE_PER_TEXT
                                        
                                        text_info = {
                                            'text': cleaned_text,
                                            'confidence': confidence,
                                            'bbox': bbox,
                                            'scored': True
                                        }
                                        detection_details['texts'].append(text_info)
                                        
            except Exception as e:
                logger.debug(f"文字检测失败 {image_path}: {e}")
            
            # 4. 根据总分数确定分类
            if total_score > self.config.REQUIRED_TOTAL_SCORE:
                category = 'high_score'
                result_msg = f'高分图片: 总分{total_score}分 (正脸{frontal_face_count}×2 + 车牌{clear_plate_count}×2 + 文字{text_count}×1)'
            elif total_score > 0:
                category = 'low_score'
                result_msg = f'低分图片: 总分{total_score}分 (正脸{frontal_face_count}×2 + 车牌{clear_plate_count}×2 + 文字{text_count}×1)'
            else:
                category = 'zero_score'
                result_msg = f'零分图片: 未检测到任何有效特征'
            
            processing_time = time.time() - start_time
            
            analysis = {
                'filename': filename,
                'category': category,
                'result': result_msg,
                'scoring': {
                    'total_score': total_score,
                    'frontal_faces': frontal_face_count,
                    'clear_plates': clear_plate_count,
                    'texts': text_count,
                    'score_breakdown': {
                        'face_score': frontal_face_count * self.config.SCORE_PER_CLEAR_FRONTAL_FACE,
                        'plate_score': clear_plate_count * self.config.SCORE_PER_CLEAR_PLATE,
                        'text_score': text_count * self.config.SCORE_PER_TEXT
                    }
                },
                'detection_details': detection_details,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            return {
                'filename': os.path.basename(image_path),
                'category': 'failed',
                'error': str(e),
                'scoring': {'total_score': 0},
                'processing_time': 0
            }
    
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

# 注释掉多进程函数，现在使用单进程模式
# def process_image_group(args):
#     """处理图片分组的工作函数（用于多进程）- 已禁用"""
#     # 该函数已被单进程批处理模式替代
#     pass

class OptimizedFacePlateDetector:
    """优化的高级人脸车牌检测器"""
    
    def __init__(self, config: Optional[OptimizedConfig] = None):
        """初始化检测器"""
        self.config = config or OptimizedConfig()
        self.start_time = time.time()
        
        # 统计信息
        self.stats = {
            'high_score': 0,       # 总分>5分的图片
            'low_score': 0,        # 1-5分的图片
            'zero_score': 0,       # 0分的图片
            'failed': 0,           # 处理失败
            'prefiltered': 0,      # 预过滤跳过
            # 保留原统计项以备兼容
            'satisfied_4_faces': 0,
            'satisfied_with_plate': 0,
            'insufficient_faces': 0,
            'no_faces': 0,
        }
        
        # 详细分析结果
        self.analysis_results = []
        
        # 进度跟踪器
        self.progress_tracker = ProgressTracker()
        
        # 获取输出目录
        self.output_dirs = self.config.get_output_dirs()
        
        # 创建输出目录
        self._create_output_dirs()
        
        logger.info("🚀 优化的高级人脸车牌检测器初始化完成")
        logger.info(f"� GPU配置: 只使用GPU 0 (避免多GPU冲突)")
        logger.info(f"�📊 新评分系统: 清晰正脸{self.config.SCORE_PER_CLEAR_FRONTAL_FACE}分 + "
                   f"清晰车牌{self.config.SCORE_PER_CLEAR_PLATE}分 + "
                   f"可识别文字{self.config.SCORE_PER_TEXT}分")
        logger.info(f"🎯 分类标准: >{self.config.REQUIRED_TOTAL_SCORE}分(符合要求) | "
                   f"1-{self.config.REQUIRED_TOTAL_SCORE}分(部分符合) | 0分(不符合)")
        logger.info(f"⚡ 启用优化: 批处理({self.config.BATCH_SIZE}), "
                   f"单GPU进程({self.config.MAX_WORKERS}), "
                   f"预过滤({self.config.PREFILTER_ENABLED})")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        for name, dir_path in self.output_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 创建目录: {dir_path}")
            except Exception as e:
                logger.error(f"❌ 创建目录失败 {dir_path}: {e}")
                raise
    
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
    
    def move_image_to_category(self, image_path: str, category: str) -> bool:
        """移动图像到对应分类目录"""
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
            
            # 删除原文件
            os.remove(image_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动图像失败 {image_path}: {e}")
            return False
    
    def save_analysis_results(self):
        """保存分析结果"""
        try:
            analysis_dir = self.output_dirs['analysis']
            
            # 保存详细分析结果
            analysis_file = os.path.join(analysis_dir, "optimized_classification_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            # 保存统计摘要
            summary = {
                'total_processed': len(self.analysis_results),
                'statistics': self.stats.copy(),
                'processing_time': time.time() - self.start_time,
                'optimization_config': {
                    'batch_size': self.config.BATCH_SIZE,
                    'max_workers': self.config.MAX_WORKERS,
                    'prefilter_enabled': self.config.PREFILTER_ENABLED,
                    'smart_grouping': self.config.SMART_GROUPING,
                },
                'timestamp': time.time()
            }
            
            summary_file = os.path.join(analysis_dir, "optimized_classification_summary.json")
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
        logger.info("🎉 基于评分系统的图片分类完成！最终统计:")
        logger.info("📊 评分规则: 清晰正脸2分 + 清晰车牌2分 + 可识别文字1分")
        logger.info("🎯 分类标准: >5分(符合要求) | 1-5分(部分符合) | 0分(不符合)")
        logger.info("")
        logger.info(f"✅ 高分图片(>5分): {self.stats['high_score']:,} 张")
        logger.info(f"📊 低分图片(1-5分): {self.stats['low_score']:,} 张")
        logger.info(f"❌ 零分图片(0分): {self.stats['zero_score']:,} 张")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,} 张")
        logger.info(f"🚫 预过滤跳过: {self.stats['prefiltered']:,} 张")
        logger.info(f"📊 总处理数量: {total_processed:,} 张")
        logger.info(f"⏰ 总耗时: {processing_time:.1f}秒")
        
        if total_processed > 0:
            avg_speed = total_processed / processing_time
            logger.info(f"🚀 平均速度: {avg_speed:.1f} 张/秒")
            
            # 计算符合要求的比例（高分图片）
            success_rate = (self.stats['high_score'] / total_processed) * 100
            logger.info(f"📈 符合要求比例: {success_rate:.1f}% (>5分)")
            
            # 计算有价值图片比例（高分+低分）
            valuable_rate = ((self.stats['high_score'] + self.stats['low_score']) / total_processed) * 100
            logger.info(f"� 有价值图片比例: {valuable_rate:.1f}% (≥1分)")
        
        # 显示各目录文件数量
        logger.info("\n📂 各分类目录统计:")
        categories = [
            ("高分图片(>5分)", self.output_dirs['high_score']),
            ("低分图片(1-5分)", self.output_dirs['low_score']),
            ("零分图片(0分)", self.output_dirs['zero_score'])
        ]
        
        for name, dir_path in categories:
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) 
                           if f.lower().endswith(tuple(self.config.SUPPORTED_FORMATS))])
                logger.info(f"  📁 {name}: {count} 张图片")
        
        logger.info("="*80)
    
    def run(self):
        """运行优化检测器"""
        logger.info("🚀 启动基于评分系统的图片分类器...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📁 输出目录: {self.config.OUTPUT_BASE_DIR}")
        logger.info(f"💻 处理模式: 单进程批处理 (避免CUDA多进程冲突)")
        logger.info(f"📊 评分规则:")
        logger.info(f"  👤 清晰正脸: {self.config.SCORE_PER_CLEAR_FRONTAL_FACE}分/张")
        logger.info(f"  🚗 清晰车牌: {self.config.SCORE_PER_CLEAR_PLATE}分/张")
        logger.info(f"  📝 可识别文字: {self.config.SCORE_PER_TEXT}分/个")
        logger.info(f"🎯 分类标准: >{self.config.REQUIRED_TOTAL_SCORE}分为符合要求")
        logger.info(f"⚡ 优化配置:")
        logger.info(f"  📦 批处理大小: {self.config.BATCH_SIZE}")
        logger.info(f"  🎯 预过滤: {'启用' if self.config.PREFILTER_ENABLED else '禁用'}")
        logger.info(f"  🧠 智能分组: {'启用' if self.config.SMART_GROUPING else '禁用'}")
        
        # 获取图像文件
        logger.info("🔍 正在扫描图像文件...")
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("❌ 未找到任何图像文件")
            return
        
        total_files = len(image_files)
        logger.info(f"📊 找到 {total_files} 张图片待处理")
        
        # 设置进度跟踪
        self.progress_tracker.add_stage("图像分组", total_files, "📦 智能分组")
        self.progress_tracker.add_stage("图像处理", total_files, "🔄 处理图像")
        self.progress_tracker.add_stage("文件移动", total_files, "📂 移动文件")
        
        # 创建分组（现在只用于批处理，不用于多进程）
        self.progress_tracker.start_stage("图像分组")
        logger.info("📊 准备批处理分组...")
        
        # 简单按文件大小排序，大文件先处理
        try:
            image_files_with_size = []
            for image_path in image_files:
                try:
                    size = os.path.getsize(image_path)
                    image_files_with_size.append((image_path, size))
                except:
                    image_files_with_size.append((image_path, 0))
            
            # 按文件大小倒序排序（大文件先处理）
            image_files_with_size.sort(key=lambda x: x[1], reverse=True)
            image_files = [item[0] for item in image_files_with_size]
            
            logger.info(f"📊 按文件大小排序完成")
        except Exception as e:
            logger.warning(f"⚠️  文件排序失败，使用原顺序: {e}")
        
        self.progress_tracker.finish_stage("图像分组")
        
        # 处理图片 - 使用单进程批处理避免CUDA多进程问题
        try:
            start_time = time.time()
            all_results = []
            
            self.progress_tracker.start_stage("图像处理")
            logger.info(f"🔄 启动单进程批处理模式...")
            
            # 初始化单个处理器
            processor = BatchProcessor(self.config)
            if not processor.initialize_models('cuda:0'):
                logger.error("❌ 模型初始化失败")
                return
            
            # 预过滤
            prefilter = PreFilter(self.config)
            filtered_images = []
            
            if self.config.PREFILTER_ENABLED:
                logger.info("🎯 开始预过滤...")
                for image_path in image_files:
                    should_process, reason = prefilter.quick_face_check(image_path)
                    if should_process:
                        filtered_images.append(image_path)
                    else:
                        self.stats['prefiltered'] += 1
                        logger.debug(f"预过滤跳过 {os.path.basename(image_path)}: {reason}")
                
                logger.info(f"📊 预过滤后剩余 {len(filtered_images)} 张图片 (跳过 {self.stats['prefiltered']} 张)")
            else:
                filtered_images = image_files
            
            # 批量处理
            grouper = ImageGrouper(self.config)
            batches = grouper.group_by_batch(filtered_images, self.config.BATCH_SIZE)
            
            logger.info(f"📦 创建 {len(batches)} 个批次，每批最多 {self.config.BATCH_SIZE} 张图片")
            
            processed_count = 0
            for batch_id, batch in enumerate(batches):
                try:
                    logger.info(f"🔄 处理批次 {batch_id + 1}/{len(batches)} ({len(batch)} 张图片)")
                    
                    batch_results = processor.process_batch(batch)
                    all_results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    # 更新进度
                    self.progress_tracker.update_stage("图像处理", processed_count)
                    
                    # 定期清理GPU缓存
                    if processor.device and 'cuda' in processor.device and (batch_id + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        logger.debug(f"🧹 清理GPU缓存 (批次 {batch_id + 1})")
                    
                    # 定期垃圾回收
                    if (batch_id + 1) % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"❌ 批次 {batch_id + 1} 处理失败: {e}")
                    # 尝试单独处理该批次中的每张图片
                    for image_path in batch:
                        try:
                            result = processor.process_single_image(image_path, None)
                            all_results.append((image_path, result))
                            processed_count += 1
                        except Exception as e2:
                            logger.error(f"❌ 单张图片处理失败 {image_path}: {e2}")
                            all_results.append((image_path, {'error': str(e2)}))
                            processed_count += 1
                    
                    self.progress_tracker.update_stage("图像处理", processed_count)
            
            self.progress_tracker.finish_stage("图像处理")
            processing_time = time.time() - start_time
            logger.info(f"🎉 单进程批处理完成，耗时 {processing_time:.1f}秒")
            
            # 处理结果
            self.progress_tracker.start_stage("文件移动")
            logger.info("📂 开始移动文件到分类目录...")
            
            moved_count = 0
            for image_path, result in all_results:
                try:
                    if 'error' in result:
                        self.stats['failed'] += 1
                        continue
                    
                    category = result.get('category', 'failed')
                    if self.move_image_to_category(image_path, category):
                        self.stats[category] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    self.analysis_results.append(result)
                    moved_count += 1
                    
                    # 更新移动进度
                    self.progress_tracker.update_stage("文件移动", moved_count)
                    
                except Exception as e:
                    logger.error(f"❌ 处理结果失败 {image_path}: {e}")
                    self.stats['failed'] += 1
            
            self.progress_tracker.finish_stage("文件移动")
        
        except KeyboardInterrupt:
            logger.info("\n⚡ 用户中断操作")
        except Exception as e:
            logger.error(f"❌ 多进程处理失败: {e}")
        
        finally:
            # 保存结果和统计
            self.progress_tracker.print_summary()
            self.save_analysis_results()
            self.print_final_statistics()

def main():
    """主函数"""
    try:
        # 创建配置
        config = OptimizedConfig()
        
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
        detector = OptimizedFacePlateDetector(config)
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
