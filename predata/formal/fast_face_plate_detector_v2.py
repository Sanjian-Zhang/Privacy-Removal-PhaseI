#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速人脸车牌检测器 - 高度优化版本
专注于速度和稳定性，移除复杂的预过滤逻辑
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
import multiprocessing as mp

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 修复CUDA多进程问题
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fast_face_plate_detector_v2.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查并导入依赖库"""
    missing_deps = []
    imported_modules = {}
    
    try:
        from retinaface import RetinaFace
        imported_modules['RetinaFace'] = RetinaFace
        logger.info("✅ RetinaFace 库导入成功")
    except ImportError as e:
        logger.error(f"❌ RetinaFace 库导入失败: {e}")
        missing_deps.append("retina-face")
    
    try:
        from ultralytics import YOLO
        imported_modules['YOLO'] = YOLO
        logger.info("✅ YOLO 库导入成功")
    except ImportError as e:
        logger.error(f"❌ YOLO 库导入失败: {e}")
        missing_deps.append("ultralytics")
    
    try:
        import easyocr
        imported_modules['easyocr'] = easyocr
        logger.info("✅ EasyOCR 库导入成功")
    except ImportError as e:
        logger.error(f"❌ EasyOCR 库导入失败: {e}")
        missing_deps.append("easyocr")
    
    try:
        import torch
        imported_modules['torch'] = torch
        logger.info("✅ PyTorch 库导入成功")
    except ImportError as e:
        logger.error(f"❌ PyTorch 库导入失败: {e}")
        missing_deps.append("torch")
    
    if missing_deps:
        logger.error(f"❌ 缺少依赖库: {', '.join(missing_deps)}")
        return None
    
    return imported_modules

# 检查并导入依赖
modules = check_dependencies()
if modules is None:
    exit(1)

RetinaFace = modules['RetinaFace']
YOLO = modules['YOLO']
easyocr = modules['easyocr']
torch = modules['torch']

class FastConfig:
    """快速检测配置类"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames70'
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames70'

    # 模型路径
    YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
    LICENSE_PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 检测阈值
    MIN_FACE_CONFIDENCE_RETINA = 0.8
    MIN_PLATE_CONFIDENCE = 0.8
    MIN_TEXT_CONFIDENCE = 0.5
    YAW_ANGLE_THRESHOLD = 30.0
    
    # 评分系统
    SCORE_PER_CLEAR_FRONTAL_FACE = 2
    SCORE_PER_CLEAR_PLATE = 2
    SCORE_PER_TEXT = 1
    REQUIRED_TOTAL_SCORE = 5
    
    # 近景判断参数 - 更严格地过滤远处人脸
    MIN_FACE_SIZE = 120                   # 提高最小人脸尺寸 (从80提高到120)
    CLOSE_UP_FACE_RATIO = 0.12            # 提高面积比例阈值 (从0.08提高到0.12)
    MIN_FACE_AREA = 14400                 # 提高最小人脸面积 (从6400提高到14400)
    MAX_DISTANCE_THRESHOLD = 0.6          # 新增：距离图片边缘的最大比例
    MIN_FACE_RESOLUTION = 150             # 新增：人脸区域的最小分辨率要求
    
    # 文件格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 优化参数（专注速度）
    BATCH_SIZE = 32                       # 大批处理大小
    ENABLE_PREFILTER = False              # 禁用预过滤
    PROGRESS_UPDATE_FREQUENCY = 100       # 降低进度更新频率
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'high_score': os.path.join(cls.OUTPUT_BASE_DIR, "high_score_images"),
            'low_score': os.path.join(cls.OUTPUT_BASE_DIR, "low_score_images"),
            'zero_score': os.path.join(cls.OUTPUT_BASE_DIR, "zero_score_images"),
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis"),
        }

class SimpleProgressBar:
    """简化的进度条"""
    
    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: Optional[int] = None):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        now = time.time()
        if now - self.last_update < 2.0 and self.current < self.total:  # 每2秒更新一次
            return
        
        self.last_update = now
        progress = self.current / self.total if self.total > 0 else 0
        percent = progress * 100
        
        elapsed = now - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
            logger.info(f"{self.prefix}: {self.current}/{self.total} ({percent:.1f}%) "
                       f"速度: {speed:.1f}/s 剩余: {eta:.0f}s")
        
        if self.current >= self.total:
            logger.info(f"✅ {self.prefix} 完成!")

class FastProcessor:
    """快速处理器"""
    
    def __init__(self, config: FastConfig):
        self.config = config
        self.device = None
        self.models = {}
        self.ocr_reader = None
        self.stats = {
            'high_score': 0,
            'low_score': 0,
            'zero_score': 0,
            'failed': 0
        }
        
    def initialize_models(self):
        """初始化模型"""
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                self.device = 'cuda:0'
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.7, device=0)
                logger.info(f"🔧 GPU初始化成功: {self.device}")
            else:
                self.device = 'cpu'
                logger.warning("⚠️  使用CPU模式")
            
            logger.info("🔄 加载YOLO人脸模型...")
            self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
            
            logger.info("🔄 加载YOLO车牌模型...")
            self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
            
            logger.info("🔄 初始化OCR...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu='cuda' in self.device)
            
            logger.info(f"✅ 模型初始化完成 ({self.device})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            try:
                self.device = 'cpu'
                self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
                self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                logger.info("✅ CPU模式初始化成功")
                return True
            except Exception as e2:
                logger.error(f"❌ CPU模式初始化也失败: {e2}")
                return False
    
    def calculate_yaw_angle(self, landmarks: Dict) -> float:
        """计算yaw角度"""
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
        except:
            return 90.0
    
    def process_single_image(self, image_path: str) -> Dict:
        """处理单张图片"""
        try:
            filename = os.path.basename(image_path)
            start_time = time.time()
            
            total_score = 0
            frontal_face_count = 0
            clear_plate_count = 0
            text_count = 0
            
            # 1. RetinaFace检测正脸
            try:
                detections = RetinaFace.detect_faces(image_path)
                
                if isinstance(detections, dict) and len(detections) > 0:
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
                            
                            # 基础尺寸过滤 - 更严格
                            if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                                continue
                            
                            if face_area < self.config.MIN_FACE_AREA:
                                continue
                            
                            # 分辨率质量检查 - 新增
                            face_resolution = max(face_width, face_height)
                            if face_resolution < self.config.MIN_FACE_RESOLUTION:
                                continue
                            
                            # 距离图片边缘检查 - 避免边缘的远景人脸
                            face_center_x = (x1 + x2) / 2
                            face_center_y = (y1 + y2) / 2
                            
                            # 计算人脸中心到图片边缘的最小距离比例
                            edge_dist_x = min(face_center_x / img_width, (img_width - face_center_x) / img_width)
                            edge_dist_y = min(face_center_y / img_height, (img_height - face_center_y) / img_height)
                            min_edge_distance = min(edge_dist_x, edge_dist_y)
                            
                            # 如果人脸太接近边缘，可能是远景，过滤掉
                            if min_edge_distance < (1 - self.config.MAX_DISTANCE_THRESHOLD):
                                continue
                            
                            # 面积比例检查 - 更严格
                            area_ratio = face_area / img_area
                            is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                            
                            # 额外的近景验证：人脸宽度占图片宽度的比例
                            width_ratio = face_width / img_width
                            height_ratio = face_height / img_height
                            size_ratio = max(width_ratio, height_ratio)
                            
                            # 只有足够大的人脸才被认为是近景
                            is_large_enough = size_ratio >= 0.15  # 人脸至少占图片尺寸的15%
                            
                            yaw_angle = self.calculate_yaw_angle(landmarks)
                            is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                            
                            # 综合判断：正面 + 近景 + 足够大 + 不在边缘
                            if is_frontal and is_close_up and is_large_enough:
                                frontal_face_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_FRONTAL_FACE
                            
            except Exception as e:
                logger.debug(f"RetinaFace检测失败 {image_path}: {e}")
            
            # 2. 检测车牌
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
                                
            except Exception as e:
                logger.debug(f"车牌检测失败 {image_path}: {e}")
            
            # 3. 检测文字
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
                                    if len(cleaned_text) >= 2:
                                        text_count += 1
                                        total_score += self.config.SCORE_PER_TEXT
                                        
            except Exception as e:
                logger.debug(f"文字检测失败 {image_path}: {e}")
            
            # 4. 确定分类
            if total_score > self.config.REQUIRED_TOTAL_SCORE:
                category = 'high_score'
            elif total_score > 0:
                category = 'low_score'
            else:
                category = 'zero_score'
            
            processing_time = time.time() - start_time
            
            return {
                'filename': filename,
                'category': category,
                'total_score': total_score,
                'frontal_faces': frontal_face_count,
                'clear_plates': clear_plate_count,
                'texts': text_count,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            return {
                'filename': os.path.basename(image_path),
                'category': 'failed',
                'error': str(e),
                'total_score': 0
            }
    
    def process_batch(self, image_paths: List[str]) -> List[Tuple[str, Dict]]:
        """批量处理"""
        results = []
        for image_path in image_paths:
            result = self.process_single_image(image_path)
            results.append((image_path, result))
        return results
    
    def move_image(self, image_path: str, category: str, output_dirs: Dict[str, str]) -> bool:
        """移动图片到分类目录"""
        try:
            filename = os.path.basename(image_path)
            output_dir = output_dirs[category]
            output_path = os.path.join(output_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
            
            shutil.copy2(image_path, output_path)
            os.remove(image_path)
            return True
            
        except Exception as e:
            logger.error(f"移动图片失败 {image_path}: {e}")
            return False
    
    def run(self):
        """运行快速检测器"""
        logger.info("🚀 启动快速人脸车牌检测器...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📦 批处理大小: {self.config.BATCH_SIZE}")
        logger.info(f"🎯 预过滤: {'启用' if self.config.ENABLE_PREFILTER else '禁用'}")
        logger.info(f"👤 人脸过滤策略: 最小尺寸{self.config.MIN_FACE_SIZE}px, 面积比例≥{self.config.CLOSE_UP_FACE_RATIO:.1%}, 最小面积{self.config.MIN_FACE_AREA}px²")
        logger.info(f"🔍 远景过滤: 最小分辨率{self.config.MIN_FACE_RESOLUTION}px, 边缘距离≥{self.config.MAX_DISTANCE_THRESHOLD:.1%}")
        
        # 初始化模型
        if not self.initialize_models():
            logger.error("❌ 模型初始化失败")
            return
        
        # 创建输出目录
        output_dirs = self.config.get_output_dirs()
        for name, dir_path in output_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
        
        # 获取图像文件
        input_path = Path(self.config.INPUT_DIR)
        image_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            image_files.extend(input_path.glob(pattern))
        
        image_files = sorted([str(f) for f in image_files if f.is_file()])
        logger.info(f"📊 找到 {len(image_files)} 张图片")
        
        if not image_files:
            logger.warning("❌ 未找到图片文件")
            return
        
        # 批量处理
        progress = SimpleProgressBar(len(image_files), "处理进度")
        processed = 0
        
        for i in range(0, len(image_files), self.config.BATCH_SIZE):
            batch = image_files[i:i + self.config.BATCH_SIZE]
            
            try:
                batch_results = self.process_batch(batch)
                
                for image_path, result in batch_results:
                    category = result.get('category', 'failed')
                    if category != 'failed':
                        if self.move_image(image_path, category, output_dirs):
                            self.stats[category] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    processed += 1
                
                progress.update(processed)
                
                # 定期清理GPU缓存
                if self.device and 'cuda' in self.device and (i // self.config.BATCH_SIZE + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                for image_path in batch:
                    self.stats['failed'] += 1
                    processed += 1
                progress.update(processed)
        
        # 打印统计结果
        logger.info("="*60)
        logger.info("🎉 处理完成！统计结果:")
        logger.info(f"✅ 高分图片(>5分): {self.stats['high_score']:,} 张")
        logger.info(f"📊 低分图片(1-5分): {self.stats['low_score']:,} 张")
        logger.info(f"❌ 零分图片(0分): {self.stats['zero_score']:,} 张")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,} 张")
        
        total = sum(self.stats.values())
        if total > 0:
            success_rate = (self.stats['high_score'] / total) * 100
            logger.info(f"📈 符合要求比例: {success_rate:.1f}%")
        
        logger.info("="*60)

def main():
    """主函数"""
    try:
        config = FastConfig()
        
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
        
        # 运行检测器
        processor = FastProcessor(config)
        processor.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
