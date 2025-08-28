#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正脸和车牌检测分类器
使用新的计分系统：
- 一张清晰正脸记2分
- 一张清晰车牌记1分
- 能够OCR识别的文字记1分
- 总分>5分认为符合要求
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import gc
from collections import defaultdict

# 第三方库导入
def check_dependencies():
    """检查依赖库"""
    missing_deps = []
    
    try:
        from retinaface import RetinaFace
        print("✅ RetinaFace 库导入成功")
    except ImportError as e:
        print(f"❌ RetinaFace 库导入失败: {e}")
        missing_deps.append("retina-face")
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO 库导入成功")
    except ImportError as e:
        print(f"❌ YOLO 库导入失败: {e}")
        missing_deps.append("ultralytics")
    
    try:
        import easyocr
        print("✅ EasyOCR 库导入成功")
    except ImportError as e:
        print(f"❌ EasyOCR 库导入失败: {e}")
        missing_deps.append("easyocr")
    
    if missing_deps:
        print(f"❌ 缺少依赖库: {', '.join(missing_deps)}")
        print("请安装缺少的库:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        return False
    
    return True

# 检查依赖
if not check_dependencies():
    exit(1)

# 正式导入
from retinaface import RetinaFace
from ultralytics import YOLO
import easyocr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 参数配置 ===
# 目录配置
INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames23/output_unique'
OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames23/output_unique/classified_frames23'
PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'

# 新计分系统阈值
SCORE_THRESHOLD = 5                # 总分阈值（>5分符合要求）
CLEAR_FACE_SCORE = 2              # 清晰正脸得分
CLEAR_PLATE_SCORE = 1             # 清晰车牌得分
TEXT_RECOGNITION_SCORE = 1        # 文字识别得分

# 检测阈值
YAW_ANGLE_THRESHOLD = 35.0
MIN_FACE_CONFIDENCE = 0.8
MIN_PLATE_CONFIDENCE = 0.5
MIN_FACE_SIZE = 60
MIN_PLATE_SIZE = 50
MIN_FACE_CLARITY_SCORE = 30.0
MAX_FACE_DISTANCE_RATIO = 0.3
FACE_AREA_THRESHOLD = 3600
MIN_TEXT_CONFIDENCE = 0.5
MIN_TEXT_LENGTH = 3

# 图像处理参数
MAX_IMAGE_SIZE = (1280, 720)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 性能参数
VERBOSE_LOGGING = True
GC_FREQUENCY = 100
PROGRESS_UPDATE_FREQUENCY = 50

# 输出目录
QUALIFIED_DIR = os.path.join(OUTPUT_BASE_DIR, "qualified")         # 符合条件(总分>5)
INSUFFICIENT_SCORE_DIR = os.path.join(OUTPUT_BASE_DIR, "insufficient_score")  # 分数不够
NO_CONTENT_DIR = os.path.join(OUTPUT_BASE_DIR, "no_content")       # 无任何有效内容
ANALYSIS_DIR = os.path.join(OUTPUT_BASE_DIR, "analysis")           # 分析结果

class FacePlateClassifier:
    """正脸和车牌检测分类器（新计分系统）"""
    
    def __init__(self):
        """初始化分类器"""
        self.start_time = time.time()
        self.stats = {
            'qualified': 0,           # 符合条件(总分>5)
            'insufficient_score': 0,  # 分数不够
            'no_content': 0,          # 无任何有效内容
            'failed': 0               # 处理失败
        }
        
        # 详细分析结果
        self.analysis_results = []
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化OCR
        self._initialize_ocr()
        
        logger.info("🚀 正脸和车牌检测分类器初始化完成（新计分系统）")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        dirs = [QUALIFIED_DIR, INSUFFICIENT_SCORE_DIR, NO_CONTENT_DIR, ANALYSIS_DIR]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"📁 创建目录: {dir_path}")
    
    def _initialize_models(self):
        """初始化检测模型"""
        try:
            # 测试RetinaFace
            logger.info("🔍 初始化RetinaFace模型...")
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            RetinaFace.detect_faces(test_img)
            logger.info("✅ RetinaFace模型初始化成功")
            
            # 初始化车牌检测模型
            logger.info("🚗 初始化车牌检测模型...")
            if not os.path.exists(PLATE_MODEL_PATH):
                raise FileNotFoundError(f"车牌检测模型文件不存在: {PLATE_MODEL_PATH}")
            
            self.plate_model = YOLO(PLATE_MODEL_PATH)
            logger.info("✅ 车牌检测模型初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def _initialize_ocr(self):
        """初始化OCR模型"""
        try:
            logger.info("📝 初始化EasyOCR模型...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])  # 支持中文简体和英文
            logger.info("✅ EasyOCR模型初始化成功")
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
            
            is_clear = clarity_score >= MIN_FACE_CLARITY_SCORE
            is_close = distance_score >= MAX_FACE_DISTANCE_RATIO
            is_large_enough = face_area >= FACE_AREA_THRESHOLD
            
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
        """使用RetinaFace检测清晰正脸"""
        try:
            detections = RetinaFace.detect_faces(image_path)
            
            if not isinstance(detections, dict) or len(detections) == 0:
                return 0, []
            
            img = cv2.imread(image_path)
            if img is None:
                return 0, []
            
            img_height, img_width = img.shape[:2]
            img_size = (img_width, img_height)
            
            clear_frontal_faces = []
            
            for face_key, face_data in detections.items():
                try:
                    confidence = face_data.get('score', 0.0)
                    if confidence < MIN_FACE_CONFIDENCE:
                        continue
                    
                    facial_area = face_data['facial_area']
                    landmarks = face_data.get('landmarks', {})
                    
                    if not landmarks:
                        continue
                    
                    x1, y1, x2, y2 = facial_area
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    if min(face_width, face_height) < MIN_FACE_SIZE:
                        continue
                    
                    # 检查人脸清晰度和距离
                    is_good_quality, quality_info = self.is_face_clear_and_close(img, facial_area, img_size)
                    
                    if not is_good_quality:
                        continue
                    
                    # 计算yaw角度
                    yaw_angle = self.calculate_yaw_angle(landmarks)
                    
                    # 判断是否为正脸
                    is_frontal = yaw_angle <= YAW_ANGLE_THRESHOLD
                    
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
            results = self.plate_model(image_path, verbose=False)
            
            if not results or len(results) == 0:
                return 0, []
            
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                return 0, []
            
            clear_plates = []
            
            for box in result.boxes:
                try:
                    confidence = float(box.conf[0])
                    if confidence < MIN_PLATE_CONFIDENCE:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    
                    if min(plate_width, plate_height) < MIN_PLATE_SIZE:
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
        """使用OCR检测可识别的文字"""
        try:
            if self.ocr_reader is None:
                return 0, []
            
            img = cv2.imread(image_path)
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
                    
                    if confidence < MIN_TEXT_CONFIDENCE:
                        continue
                    
                    cleaned_text = text.strip()
                    if len(cleaned_text) < MIN_TEXT_LENGTH:
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
    
    def classify_image(self, image_path: str) -> Tuple[str, Dict]:
        """使用新计分系统分类图像"""
        try:
            filename = os.path.basename(image_path)
            
            # 检测清晰正脸
            frontal_count, face_details = self.detect_faces(image_path)
            
            # 检测清晰车牌
            plate_count, plate_details = self.detect_license_plates(image_path)
            
            # 检测可识别文字
            text_count, text_details = self.detect_text(image_path)
            
            # 新计分系统
            score = 0
            score_details = []
            
            # 清晰正脸：每张2分
            if frontal_count > 0:
                face_score = frontal_count * CLEAR_FACE_SCORE
                score += face_score
                score_details.append(f"清晰正脸 {frontal_count} 张 × {CLEAR_FACE_SCORE} = {face_score} 分")
            
            # 清晰车牌：每张1分
            if plate_count > 0:
                plate_score = plate_count * CLEAR_PLATE_SCORE
                score += plate_score
                score_details.append(f"清晰车牌 {plate_count} 张 × {CLEAR_PLATE_SCORE} = {plate_score} 分")
            
            # 可识别文字：1分（无论多少段）
            if text_count > 0:
                text_score = TEXT_RECOGNITION_SCORE
                score += text_score
                score_details.append(f"可识别文字 = {text_score} 分")
            
            # 判断是否符合要求（总分>5）
            meets_requirements = score > SCORE_THRESHOLD
            
            # 创建分析结果
            analysis = {
                'filename': filename,
                'frontal_faces': frontal_count,
                'license_plates': plate_count,
                'text_count': text_count,
                'total_score': score,
                'score_details': score_details,
                'meets_requirements': meets_requirements,
                'score_threshold': SCORE_THRESHOLD,
                'face_details': face_details,
                'plate_details': plate_details,
                'text_details': text_details,
                'timestamp': time.time()
            }
            
            # 分类逻辑
            if meets_requirements:
                category = 'qualified'
                analysis['qualification_reason'] = f'总分 {score} 分 > {SCORE_THRESHOLD} 分，符合要求'
            else:
                if score == 0:
                    category = 'no_content'
                    analysis['reject_reason'] = f'总分 {score} 分，无任何有效内容'
                else:
                    category = 'insufficient_score'
                    analysis['reject_reason'] = f'总分 {score} 分 ≤ {SCORE_THRESHOLD} 分，不符合要求'
            
            analysis['category'] = category
            
            return category, analysis
            
        except Exception as e:
            logger.error(f"❌ 图像分类失败 {image_path}: {e}")
            return 'failed', {'filename': os.path.basename(image_path), 'error': str(e)}
    
    def copy_image_to_category(self, image_path: str, category: str) -> bool:
        """复制图像到对应分类目录"""
        try:
            filename = os.path.basename(image_path)
            
            category_dirs = {
                'qualified': QUALIFIED_DIR,
                'insufficient_score': INSUFFICIENT_SCORE_DIR,
                'no_content': NO_CONTENT_DIR
            }
            
            if category not in category_dirs:
                return False
            
            output_dir = category_dirs[category]
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
        input_path = Path(INPUT_DIR)
        
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {INPUT_DIR}")
            return []
        
        logger.info(f"🔍 扫描目录: {INPUT_DIR}")
        
        for ext in SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            files.extend(input_path.glob(pattern))
        
        image_files = sorted([str(f) for f in files if f.is_file()])
        logger.info(f"📊 找到 {len(image_files)} 个图像文件")
        
        return image_files
    
    def save_analysis_results(self):
        """保存分析结果"""
        try:
            # 保存详细分析结果
            analysis_file = os.path.join(ANALYSIS_DIR, "classification_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            # 保存统计摘要
            summary = {
                'total_processed': len(self.analysis_results),
                'statistics': self.stats.copy(),
                'processing_time': time.time() - self.start_time,
                'scoring_system': {
                    'clear_face_score': CLEAR_FACE_SCORE,
                    'clear_plate_score': CLEAR_PLATE_SCORE,
                    'text_recognition_score': TEXT_RECOGNITION_SCORE,
                    'score_threshold': SCORE_THRESHOLD
                },
                'configuration': {
                    'yaw_angle_threshold': YAW_ANGLE_THRESHOLD,
                    'min_face_confidence': MIN_FACE_CONFIDENCE,
                    'min_plate_confidence': MIN_PLATE_CONFIDENCE,
                    'min_text_confidence': MIN_TEXT_CONFIDENCE
                },
                'timestamp': time.time()
            }
            
            summary_file = os.path.join(ANALYSIS_DIR, "classification_summary.json")
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
        logger.info("🎉 正脸和车牌分类完成！最终统计:")
        logger.info("新的计分系统:")
        logger.info(f"  - 一张清晰正脸 = {CLEAR_FACE_SCORE} 分")
        logger.info(f"  - 一张清晰车牌 = {CLEAR_PLATE_SCORE} 分")  
        logger.info(f"  - 能识别的文字 = {TEXT_RECOGNITION_SCORE} 分")
        logger.info(f"  - 总分 > {SCORE_THRESHOLD} 分 = 符合要求")
        logger.info(f"✅ 符合条件(>{SCORE_THRESHOLD}分): {self.stats['qualified']:,}")
        logger.info(f"❌ 分数不够(≤{SCORE_THRESHOLD}分): {self.stats['insufficient_score']:,}")
        logger.info(f"❌ 无任何内容: {self.stats['no_content']:,}")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,}")
        logger.info(f"📊 总处理数量: {total_processed:,}")
        logger.info(f"⏰ 总耗时: {processing_time:.1f}秒")
        
        if total_processed > 0:
            avg_speed = total_processed / processing_time
            logger.info(f"🚀 平均速度: {avg_speed:.1f} 张/秒")
            
            success_rate = (self.stats['qualified'] / total_processed) * 100
            logger.info(f"📈 符合条件比例: {success_rate:.1f}%")
        
        # 显示各目录文件数量
        logger.info("\n📂 各分类目录统计:")
        categories = [
            ("符合条件", QUALIFIED_DIR),
            ("分数不够", INSUFFICIENT_SCORE_DIR),
            ("无任何内容", NO_CONTENT_DIR)
        ]
        
        for name, dir_path in categories:
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) 
                           if f.lower().endswith(tuple(SUPPORTED_FORMATS))])
                logger.info(f"  📁 {name}: {count} 张图片")
        
        logger.info("="*80)
    
    def run(self):
        """运行分类器"""
        logger.info("🚀 启动正脸和车牌检测分类器（新计分系统）...")
        logger.info(f"📁 输入目录: {INPUT_DIR}")
        logger.info(f"📁 输出目录: {OUTPUT_BASE_DIR}")
        logger.info(f"📊 计分规则:")
        logger.info(f"  - 清晰正脸: {CLEAR_FACE_SCORE} 分/张")
        logger.info(f"  - 清晰车牌: {CLEAR_PLATE_SCORE} 分/张")
        logger.info(f"  - 可识别文字: {TEXT_RECOGNITION_SCORE} 分")
        logger.info(f"  - 通过阈值: > {SCORE_THRESHOLD} 分")
        logger.info(f"📐 yaw角度阈值: {YAW_ANGLE_THRESHOLD}°")
        logger.info(f"🎯 人脸置信度阈值: {MIN_FACE_CONFIDENCE}")
        logger.info(f"🎯 车牌置信度阈值: {MIN_PLATE_CONFIDENCE}")
        logger.info(f"🎯 文字识别置信度阈值: {MIN_TEXT_CONFIDENCE}")
        logger.info(f"🔍 最小清晰度分数: {MIN_FACE_CLARITY_SCORE}")
        logger.info(f"📏 最小人脸面积: {FACE_AREA_THRESHOLD}px²")
        
        # 获取图像文件
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("❌ 未找到任何图像文件")
            return
        
        # 开始处理
        try:
            with tqdm(
                total=len(image_files),
                desc="分类进度",
                ncols=120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ) as pbar:
                
                for i, image_path in enumerate(image_files):
                    try:
                        # 分类图像
                        category, analysis = self.classify_image(image_path)
                        
                        if category != 'failed':
                            # 复制到对应目录
                            if self.copy_image_to_category(image_path, category):
                                self.stats[category] += 1
                            else:
                                self.stats['failed'] += 1
                                analysis['copy_failed'] = True
                        else:
                            self.stats['failed'] += 1
                        
                        # 保存分析结果
                        self.analysis_results.append(analysis)
                        
                        # 更新进度条
                        pbar.update(1)
                        
                        # 更新描述
                        if i % PROGRESS_UPDATE_FREQUENCY == 0 and i > 0:
                            stats_str = f"✅{self.stats['qualified']} ❌{self.stats['insufficient_score']} ⭕{self.stats['no_content']}"
                            pbar.set_description(f"分类进度 ({stats_str})")
                        
                        # 定期内存清理
                        if i % GC_FREQUENCY == 0 and i > 0:
                            gc.collect()
                    
                    except Exception as e:
                        logger.error(f"❌ 处理图像失败 {image_path}: {e}")
                        self.stats['failed'] += 1
                        pbar.update(1)
        
        finally:
            # 保存结果和统计
            self.save_analysis_results()
            self.print_final_statistics()

def main():
    """主函数"""
    try:
        # 检查输入目录
        if not os.path.exists(INPUT_DIR):
            logger.error(f"❌ 输入目录不存在: {INPUT_DIR}")
            return
        
        # 检查车牌检测模型
        if not os.path.exists(PLATE_MODEL_PATH):
            logger.error(f"❌ 车牌检测模型不存在: {PLATE_MODEL_PATH}")
            return
        
        # 创建分类器并运行
        classifier = FacePlateClassifier()
        classifier.run()
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
