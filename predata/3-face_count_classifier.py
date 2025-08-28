#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正脸数量分类器
使用YOLOv8s和RetinaFace检测正脸数量，按以下规则分类：
- 0张正脸
- 1-2张正脸  
- 3-6张正脸
- 6-9张正脸
- 9-12张正脸
- 12+张正脸
"""

import os
import cv2
import numpy as np
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
        logging.FileHandler('face_count_classifier.log', encoding='utf-8')
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
torch = modules['torch']

class FaceCountConfig:
    """正脸数量分类配置"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames70/high_score_images_blurred/0_faces/0_faces/0_faces'
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames70'

    # 模型路径
    YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
    
    # 检测阈值
    MIN_FACE_CONFIDENCE_RETINA = 0.9    # RetinaFace最小置信度
    YAW_ANGLE_THRESHOLD = 25.0           # yaw角度阈值（正脸）
    MIN_FACE_SIZE = 65                   # 最小人脸尺寸（像素）- 提高以忽略远处的脸
    MIN_FACE_AREA = 4500                 # 最小人脸面积（80x80）- 提高以忽略远处的脸
    MIN_FACE_SIZE_RATIO = 0.0005           # 人脸相对于图像的最小比例
    MAX_DISTANCE_THRESHOLD = 0.15         # 基于图像大小的最大距离阈值
    # 文件格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 处理参数
    BATCH_SIZE = 16                      # 批处理大小
    PROGRESS_UPDATE_FREQUENCY = 50       # 进度更新频率
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'no_faces': os.path.join(cls.OUTPUT_BASE_DIR, "0_faces"),           # 0张正脸
            'faces_1_2': os.path.join(cls.OUTPUT_BASE_DIR, "1-2_faces"),       # 1-2张正脸
            'faces_3_6': os.path.join(cls.OUTPUT_BASE_DIR, "3-6_faces"),       # 3-6张正脸
            'faces_6_9': os.path.join(cls.OUTPUT_BASE_DIR, "6-9_faces"),       # 6-9张正脸
            'faces_9_12': os.path.join(cls.OUTPUT_BASE_DIR, "9-12_faces"),     # 9-12张正脸
            'faces_12_plus': os.path.join(cls.OUTPUT_BASE_DIR, "12+_faces"),   # 12+张正脸
            'failed': os.path.join(cls.OUTPUT_BASE_DIR, "failed"),             # 处理失败
        }

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, name: str = "处理进度"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: Optional[int] = None):
        """更新进度"""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        now = time.time()
        if now - self.last_update < 3.0 and self.current < self.total:  # 每3秒更新一次
            return
        
        self.last_update = now
        progress = self.current / self.total if self.total > 0 else 0
        percent = progress * 100
        
        elapsed = now - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
            logger.info(f"📊 {self.name}: {self.current}/{self.total} ({percent:.1f}%) "
                       f"速度: {speed:.1f}/s 预计剩余: {eta:.0f}s")
        
        if self.current >= self.total:
            elapsed_total = time.time() - self.start_time
            logger.info(f"✅ {self.name} 完成! 总耗时: {elapsed_total:.1f}s")

class FaceCountClassifier:
    """正脸数量分类器"""
    
    def __init__(self, config: FaceCountConfig):
        self.config = config
        self.device = None
        self.yolo_model = None
        self.stats = {
            'no_faces': 0,
            'faces_1_2': 0,
            'faces_3_6': 0,
            'faces_6_9': 0,
            'faces_9_12': 0,
            'faces_12_plus': 0,
            'failed': 0
        }
        
        # 获取输出目录
        self.output_dirs = self.config.get_output_dirs()
        self._create_output_dirs()
        
    def _create_output_dirs(self):
        """创建输出目录"""
        for name, dir_path in self.output_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 创建目录: {dir_path}")
            except Exception as e:
                logger.error(f"❌ 创建目录失败 {dir_path}: {e}")
                raise
    
    def initialize_models(self):
        """初始化模型"""
        try:
            # 初始化GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                self.device = 'cuda:0'
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.7, device=0)
                logger.info(f"🔧 GPU初始化成功: {self.device}")
            else:
                self.device = 'cpu'
                logger.warning("⚠️  使用CPU模式")
            
            # 加载YOLO模型（用于人脸区域粗检测）
            logger.info("🔄 加载YOLOv8s模型...")
            self.yolo_model = YOLO(self.config.YOLOV8S_MODEL_PATH)
            
            logger.info(f"✅ 模型初始化完成 ({self.device})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            try:
                # 尝试CPU模式
                self.device = 'cpu'
                self.yolo_model = YOLO(self.config.YOLOV8S_MODEL_PATH)
                logger.info("✅ CPU模式初始化成功")
                return True
            except Exception as e2:
                logger.error(f"❌ CPU模式初始化也失败: {e2}")
                return False
    
    def calculate_yaw_angle(self, landmarks: Dict) -> float:
        """基于RetinaFace关键点计算yaw角度"""
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            
            # 计算眼睛中心点
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width < 10:  # 眼睛距离太小，可能是侧脸
                return 90.0
            
            # 计算鼻子相对于眼睛中心的水平偏移
            horizontal_offset = nose[0] - eye_center[0]
            normalized_offset = horizontal_offset / eye_width
            
            # 将偏移转换为角度估计
            yaw_angle = abs(normalized_offset) * 60.0  # 经验公式
            
            return yaw_angle
            
        except Exception as e:
            logger.debug(f"yaw角度计算失败: {e}")
            return 90.0  # 返回大角度，表示非正脸
    
    def count_frontal_faces(self, image_path: str) -> Tuple[int, str]:
        """统计正脸数量"""
        try:
            filename = os.path.basename(image_path)
            
            # 使用RetinaFace检测人脸
            detections = RetinaFace.detect_faces(image_path)
            
            if not isinstance(detections, dict) or len(detections) == 0:
                return 0, "未检测到人脸"
            
            # 读取图像信息
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return 0, "无法读取图片"
            
            img_height, img_width = img.shape[:2]
            img_area = img_height * img_width
            frontal_face_count = 0
            
            for face_key, face_data in detections.items():
                try:
                    # 检查置信度
                    confidence = face_data.get('score', 0.0)
                    if confidence < self.config.MIN_FACE_CONFIDENCE_RETINA:
                        continue
                    
                    # 获取人脸区域和关键点
                    facial_area = face_data['facial_area']
                    landmarks = face_data.get('landmarks', {})
                    
                    if not landmarks:
                        continue
                    
                    # 检查人脸大小
                    x1, y1, x2, y2 = facial_area
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_area = face_width * face_height
                    
                    # 过滤太小的人脸（基本尺寸）
                    if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                        continue
                    
                    if face_area < self.config.MIN_FACE_AREA:
                        continue
                    
                    # 过滤相对于图像太小的人脸（远距离人脸）
                    face_size_ratio = face_area / img_area
                    if face_size_ratio < self.config.MIN_FACE_SIZE_RATIO:
                        continue
                    
                    # 基于图像对角线的距离过滤
                    img_diagonal = np.sqrt(img_width**2 + img_height**2)
                    face_diagonal = np.sqrt(face_width**2 + face_height**2)
                    distance_ratio = face_diagonal / img_diagonal
                    if distance_ratio < self.config.MAX_DISTANCE_THRESHOLD:
                        continue
                    
                    # 额外的远距离过滤：检查人脸是否在图像边缘附近（可能是背景中的小人脸）
                    face_center_x = (x1 + x2) / 2
                    face_center_y = (y1 + y2) / 2
                    
                    # 如果人脸在图像边缘且很小，可能是远处的人脸
                    edge_threshold = 0.1  # 10%边缘区域
                    is_near_edge = (face_center_x < img_width * edge_threshold or 
                                   face_center_x > img_width * (1 - edge_threshold) or
                                   face_center_y < img_height * edge_threshold or 
                                   face_center_y > img_height * (1 - edge_threshold))
                    
                    if is_near_edge and face_size_ratio < 0.05:  # 边缘且很小的人脸
                        continue
                    
                    # 计算yaw角度判断是否为正脸
                    yaw_angle = self.calculate_yaw_angle(landmarks)
                    is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                    
                    if is_frontal:
                        frontal_face_count += 1
                        
                except Exception as e:
                    logger.debug(f"处理单个人脸失败 {filename}: {e}")
                    continue
            
            return frontal_face_count, f"检测到{frontal_face_count}张正脸"
            
        except Exception as e:
            logger.error(f"正脸检测失败 {image_path}: {e}")
            return -1, f"处理失败: {str(e)}"
    
    def classify_by_face_count(self, face_count: int) -> str:
        """根据正脸数量确定分类"""
        if face_count == 0:
            return 'no_faces'
        elif 1 <= face_count <= 2:
            return 'faces_1_2'
        elif 3 <= face_count <= 6:
            return 'faces_3_6'
        elif 6 <= face_count <= 9:
            return 'faces_6_9'
        elif 9 <= face_count <= 12:
            return 'faces_9_12'
        else:  # 12+
            return 'faces_12_plus'
    
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
            
            # 复制文件并删除原文件
            shutil.copy2(image_path, output_path)
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
    
    def process_batch(self, image_batch: List[str]) -> List[Tuple[str, int, str]]:
        """批量处理图片"""
        results = []
        for image_path in image_batch:
            face_count, message = self.count_frontal_faces(image_path)
            results.append((image_path, face_count, message))
        return results
    
    def run(self):
        """运行正脸数量分类器"""
        logger.info("="*80)
        logger.info("🚀 正脸数量分类器启动")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📁 输出目录: {self.config.OUTPUT_BASE_DIR}")
        logger.info(f"📊 分类规则:")
        logger.info(f"  📂 0张正脸 → 0_faces")
        logger.info(f"  📂 1-2张正脸 → 1-2_faces")
        logger.info(f"  📂 3-6张正脸 → 3-6_faces")
        logger.info(f"  📂 6-9张正脸 → 6-9_faces")
        logger.info(f"  📂 9-12张正脸 → 9-12_faces")
        logger.info(f"  📂 12+张正脸 → 12+_faces")
        logger.info(f"⚙️  RetinaFace置信度阈值: {self.config.MIN_FACE_CONFIDENCE_RETINA}")
        logger.info(f"⚙️  正脸角度阈值: {self.config.YAW_ANGLE_THRESHOLD}°")
        logger.info(f"⚙️  最小人脸尺寸: {self.config.MIN_FACE_SIZE}px")
        logger.info(f"⚙️  最小人脸面积: {self.config.MIN_FACE_AREA}px²")
        logger.info(f"⚙️  最小人脸比例: {self.config.MIN_FACE_SIZE_RATIO:.3f}")
        logger.info(f"⚙️  距离过滤阈值: {self.config.MAX_DISTANCE_THRESHOLD:.2f}")
        logger.info("="*80)
        
        # 初始化模型
        if not self.initialize_models():
            logger.error("❌ 模型初始化失败")
            return
        
        # 获取图像文件
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("❌ 未找到任何图像文件")
            return
        
        # 处理图片
        total_files = len(image_files)
        progress = ProgressTracker(total_files, "分类进度")
        
        processed = 0
        for i in range(0, total_files, self.config.BATCH_SIZE):
            batch = image_files[i:i + self.config.BATCH_SIZE]
            
            try:
                batch_results = self.process_batch(batch)
                
                for image_path, face_count, message in batch_results:
                    try:
                        if face_count >= 0:  # 处理成功
                            category = self.classify_by_face_count(face_count)
                            if self.move_image_to_category(image_path, category):
                                self.stats[category] += 1
                                logger.debug(f"✅ {os.path.basename(image_path)}: {face_count}张正脸 → {category}")
                            else:
                                self.stats['failed'] += 1
                        else:  # 处理失败
                            if self.move_image_to_category(image_path, 'failed'):
                                self.stats['failed'] += 1
                            else:
                                logger.error(f"❌ 无法移动失败文件: {image_path}")
                        
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"❌ 处理结果失败 {image_path}: {e}")
                        self.stats['failed'] += 1
                        processed += 1
                
                progress.update(processed)
                
                # 定期清理GPU缓存
                if self.device and 'cuda' in self.device and (i // self.config.BATCH_SIZE + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"❌ 批次处理失败: {e}")
                # 将批次中的所有文件标记为失败
                for image_path in batch:
                    self.stats['failed'] += 1
                    processed += 1
                progress.update(processed)
        
        # 打印最终统计
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        total_processed = sum(self.stats.values())
        
        logger.info("="*80)
        logger.info("🎉 正脸数量分类完成！最终统计:")
        logger.info(f"📊 总处理图片数: {total_processed:,} 张")
        logger.info("")
        logger.info("📂 各分类统计:")
        logger.info(f"  📁 0张正脸: {self.stats['no_faces']:,} 张")
        logger.info(f"  📁 1-2张正脸: {self.stats['faces_1_2']:,} 张")
        logger.info(f"  📁 3-6张正脸: {self.stats['faces_3_6']:,} 张")
        logger.info(f"  📁 6-9张正脸: {self.stats['faces_6_9']:,} 张")
        logger.info(f"  📁 9-12张正脸: {self.stats['faces_9_12']:,} 张")
        logger.info(f"  📁 12+张正脸: {self.stats['faces_12_plus']:,} 张")
        logger.info(f"  ❌ 处理失败: {self.stats['failed']:,} 张")
        
        if total_processed > 0:
            # 计算有效图片比例（有人脸的图片）
            valid_faces = total_processed - self.stats['no_faces'] - self.stats['failed']
            valid_rate = (valid_faces / total_processed) * 100
            logger.info(f"📈 有人脸图片比例: {valid_rate:.1f}% ({valid_faces:,}/{total_processed:,})")
            
            # 计算多人脸图片比例（3张以上）
            multi_faces = (self.stats['faces_3_6'] + self.stats['faces_6_9'] + 
                          self.stats['faces_9_12'] + self.stats['faces_12_plus'])
            multi_rate = (multi_faces / total_processed) * 100
            logger.info(f"👥 多人脸图片比例: {multi_rate:.1f}% ({multi_faces:,}/{total_processed:,})")
        
        logger.info("\n📂 各分类目录:")
        for category, count in self.stats.items():
            if count > 0:
                dir_path = self.output_dirs[category]
                logger.info(f"  {dir_path}: {count:,} 张")
        
        logger.info("="*80)

def main():
    """主函数"""
    try:
        config = FaceCountConfig()
        
        # 检查输入目录
        if not os.path.exists(config.INPUT_DIR):
            logger.error(f"❌ 输入目录不存在: {config.INPUT_DIR}")
            return
        
        # 检查模型文件
        if not os.path.exists(config.YOLOV8S_MODEL_PATH):
            logger.error(f"❌ YOLOv8s模型不存在: {config.YOLOV8S_MODEL_PATH}")
            return
        
        # 创建分类器并运行
        classifier = FaceCountClassifier(config)
        classifier.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
