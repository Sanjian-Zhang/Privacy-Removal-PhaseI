#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分系统测试脚本 - CPU稳定版
用于测试新的评分机制：
- 清晰正脸：2分/张
- 清晰车牌：2分/张  
- 可识别文字：1分/个
- >5分：符合要求
- 1-5分：部分符合
- 0分：不符合
"""

import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# 强制使用CPU模式，避免GPU问题
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入依赖
try:
    from ultralytics import YOLO
    from retinaface import RetinaFace
    import easyocr
    import torch
    logger.info("✅ 所有依赖库加载成功 (CPU模式)")
except ImportError as e:
    logger.error(f"❌ 依赖库加载失败: {e}")
    exit(1)

class ScoringConfig:
    """评分系统配置"""
    
    # 目录配置
    INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames69'
    OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames69/scored'

    # 模型路径
    YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
    LICENSE_PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
    
    # 评分规则
    SCORE_PER_CLEAR_FRONTAL_FACE = 2
    SCORE_PER_CLEAR_PLATE = 2
    SCORE_PER_TEXT = 1
    REQUIRED_TOTAL_SCORE = 5
    
    # 检测阈值
    MIN_FACE_CONFIDENCE_RETINA = 0.7  # 降低一点阈值
    MIN_PLATE_CONFIDENCE = 0.4        # 降低一点阈值
    MIN_TEXT_CONFIDENCE = 0.4         # 降低一点阈值
    YAW_ANGLE_THRESHOLD = 35.0        # 放宽一点角度
    
    # 近景判断
    MIN_FACE_SIZE = 60                # 降低最小尺寸
    CLOSE_UP_FACE_RATIO = 0.06        # 降低面积比例
    MIN_FACE_AREA = 3600              # 降低最小面积(60x60)
    
    # 文件格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    @classmethod
    def get_output_dirs(cls):
        """获取输出目录配置"""
        return {
            'high_score': os.path.join(cls.OUTPUT_BASE_DIR, "high_score_images"),  # >5分
            'low_score': os.path.join(cls.OUTPUT_BASE_DIR, "low_score_images"),   # 1-5分
            'zero_score': os.path.join(cls.OUTPUT_BASE_DIR, "zero_score_images"), # 0分
            'analysis': os.path.join(cls.OUTPUT_BASE_DIR, "analysis"),
        }

class StableImageScorer:
    """稳定的图片评分器 - CPU版本"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.device = 'cpu'  # 强制使用CPU
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 初始化模型
        self._initialize_models()
        
        logger.info(f"🚀 稳定图片评分器初始化完成 (CPU模式)")
        logger.info(f"📊 评分规则: 正脸{config.SCORE_PER_CLEAR_FRONTAL_FACE}分 + "
                   f"车牌{config.SCORE_PER_CLEAR_PLATE}分 + 文字{config.SCORE_PER_TEXT}分")
        logger.info(f"🎯 符合要求标准: >{config.REQUIRED_TOTAL_SCORE}分")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        output_dirs = self.config.get_output_dirs()
        for name, dir_path in output_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"📁 创建目录: {dir_path}")
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            # 初始化车牌检测模型 (CPU模式)
            logger.info("🔧 加载车牌检测模型...")
            self.plate_model = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
            logger.info("✅ 车牌检测模型加载成功")
            
            # 初始化OCR (CPU模式)
            logger.info("🔧 加载OCR模型...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            logger.info("✅ OCR模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def calculate_yaw_angle(self, landmarks: dict) -> float:
        """计算yaw角度"""
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width < 5:  # 降低阈值
                return 90.0
            
            horizontal_offset = nose[0] - eye_center[0]
            normalized_offset = horizontal_offset / eye_width
            yaw_angle = abs(normalized_offset) * 60.0
            
            return yaw_angle
            
        except Exception:
            return 90.0
    
    def score_image(self, image_path: str) -> dict:
        """对单张图片进行评分"""
        try:
            start_time = time.time()
            filename = os.path.basename(image_path)
            
            # 初始化分数
            total_score = 0
            frontal_face_count = 0
            clear_plate_count = 0
            text_count = 0
            
            details = {
                'faces': [],
                'plates': [],
                'texts': []
            }
            
            logger.info(f"🔍 评分图片: {filename}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 1. 检测清晰正脸 (RetinaFace) - 使用安全模式
            try:
                logger.info("  👤 检测正脸...")
                
                # 先读取图片检查是否有效
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("无法读取图片")
                
                img_height, img_width = img.shape[:2]
                img_area = img_width * img_height
                
                # 使用RetinaFace检测
                detections = RetinaFace.detect_faces(image_path)
                
                if isinstance(detections, dict) and len(detections) > 0:
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
                            
                            # 检查尺寸
                            if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                                continue
                            if face_area < self.config.MIN_FACE_AREA:
                                continue
                            
                            # 检查是否为正脸
                            yaw_angle = self.calculate_yaw_angle(landmarks)
                            is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                            
                            # 检查是否为近景
                            area_ratio = face_area / img_area
                            is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                            
                            face_info = {
                                'confidence': confidence,
                                'yaw_angle': yaw_angle,
                                'is_frontal': is_frontal,
                                'is_close_up': is_close_up,
                                'area_ratio': area_ratio
                            }
                            details['faces'].append(face_info)
                            
                            # 只有清晰的近景正脸才计分
                            if is_frontal and is_close_up:
                                frontal_face_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_FRONTAL_FACE
                                logger.info(f"    ✅ 清晰正脸 #{frontal_face_count} (置信度: {confidence:.2f}, yaw: {yaw_angle:.1f}°, 面积比: {area_ratio:.3f})")
                            else:
                                reason = []
                                if not is_frontal:
                                    reason.append(f"非正脸(yaw={yaw_angle:.1f}°)")
                                if not is_close_up:
                                    reason.append(f"非近景(比例={area_ratio:.3f})")
                                logger.info(f"    ⚠️  人脸未计分: {', '.join(reason)}")
                        
                        except Exception as e:
                            logger.debug(f"    处理单个人脸失败: {e}")
                            continue
                
                logger.info(f"  👤 正脸检测结果: {frontal_face_count} 张 -> {frontal_face_count * self.config.SCORE_PER_CLEAR_FRONTAL_FACE} 分")
                        
            except Exception as e:
                logger.warning(f"  RetinaFace检测失败: {e}")
            
            # 2. 检测清晰车牌
            try:
                logger.info("  🚗 检测车牌...")
                results = self.plate_model(image_path, verbose=False, device=self.device)
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        for i, box in enumerate(result.boxes):
                            confidence = float(box.conf[0])
                            if confidence >= self.config.MIN_PLATE_CONFIDENCE:
                                clear_plate_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_PLATE
                                
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                plate_info = {
                                    'confidence': confidence,
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                }
                                details['plates'].append(plate_info)
                                
                                logger.info(f"    ✅ 清晰车牌 #{clear_plate_count} (置信度: {confidence:.2f})")
                            else:
                                logger.info(f"    ⚠️  车牌置信度不足: {confidence:.2f}")
                
                logger.info(f"  🚗 车牌检测结果: {clear_plate_count} 张 -> {clear_plate_count * self.config.SCORE_PER_CLEAR_PLATE} 分")
                        
            except Exception as e:
                logger.warning(f"  车牌检测失败: {e}")
            
            # 3. 检测可识别文字
            try:
                logger.info("  📝 检测文字...")
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is not None:
                    ocr_results = self.ocr_reader.readtext(img)
                    
                    valid_texts = []
                    if ocr_results:
                        for bbox, text, confidence in ocr_results:
                            confidence = float(confidence) if confidence is not None else 0.0
                            
                            if confidence >= self.config.MIN_TEXT_CONFIDENCE:
                                cleaned_text = text.strip()
                                if len(cleaned_text) >= 2:
                                    text_count += 1
                                    total_score += self.config.SCORE_PER_TEXT
                                    
                                    text_info = {
                                        'text': cleaned_text,
                                        'confidence': confidence,
                                        'bbox': bbox
                                    }
                                    details['texts'].append(text_info)
                                    
                                    valid_texts.append(f"'{cleaned_text}'({confidence:.2f})")
                            else:
                                logger.debug(f"    文字置信度不足: '{text}' ({confidence:.2f})")
                    
                    if valid_texts:
                        logger.info(f"    ✅ 可识别文字: {', '.join(valid_texts)}")
                    else:
                        logger.info(f"    ⚠️  未检测到高置信度文字")
                    
                    logger.info(f"  📝 文字检测结果: {text_count} 个 -> {text_count * self.config.SCORE_PER_TEXT} 分")
                        
            except Exception as e:
                logger.warning(f"  文字检测失败: {e}")
            
            # 4. 计算总分和分类
            processing_time = time.time() - start_time
            
            if total_score > self.config.REQUIRED_TOTAL_SCORE:
                category = 'high_score'
                status = '✅ 符合要求'
                color = '🟢'
            elif total_score > 0:
                category = 'low_score'
                status = '⚠️  部分符合'
                color = '🟡'
            else:
                category = 'zero_score'
                status = '❌ 不符合要求'
                color = '🔴'
            
            logger.info(f"  📊 {color} 总分: {total_score} 分 -> {status}")
            logger.info(f"  ⏰ 处理耗时: {processing_time:.2f}秒")
            logger.info("")
            
            return {
                'filename': filename,
                'category': category,
                'status': status,
                'total_score': total_score,
                'frontal_faces': frontal_face_count,
                'clear_plates': clear_plate_count,
                'texts': text_count,
                'score_breakdown': {
                    'face_score': frontal_face_count * self.config.SCORE_PER_CLEAR_FRONTAL_FACE,
                    'plate_score': clear_plate_count * self.config.SCORE_PER_CLEAR_PLATE,
                    'text_score': text_count * self.config.SCORE_PER_TEXT
                },
                'details': details,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ 评分失败 {image_path}: {e}")
            return {
                'filename': os.path.basename(image_path),
                'category': 'failed',
                'error': str(e),
                'total_score': 0,
                'processing_time': 0
            }
    
    def move_image_to_category(self, image_path: str, category: str) -> bool:
        """移动图像到分类目录"""
        try:
            filename = os.path.basename(image_path)
            output_dirs = self.config.get_output_dirs()
            
            if category not in output_dirs:
                logger.error(f"❌ 未知分类: {category}")
                return False
            
            output_dir = output_dirs[category]
            output_path = os.path.join(output_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
            
            # 复制文件
            import shutil
            shutil.copy2(image_path, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动图像失败 {image_path}: {e}")
            return False
    
    def process_images(self, max_images: int | None = None, move_files: bool = False):
        """处理图片"""
        # 获取图片文件
        input_path = Path(self.config.INPUT_DIR)
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {self.config.INPUT_DIR}")
            return
        
        image_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        image_files = sorted([str(f) for f in image_files if f.is_file()])
        
        if not image_files:
            logger.warning("❌ 未找到图像文件")
            return
        
        # 限制处理数量
        if max_images and max_images > 0:
            image_files = image_files[:max_images]
        
        logger.info(f"🧪 开始处理图片，数量: {len(image_files)}")
        logger.info("="*80)
        
        # 处理每张图片
        results = []
        stats = {'high_score': 0, 'low_score': 0, 'zero_score': 0, 'failed': 0}
        start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📸 处理图片 {i}/{len(image_files)}")
            result = self.score_image(image_path)
            results.append(result)
            
            category = result.get('category', 'failed')
            if category in stats:
                stats[category] += 1
            
            # 移动文件（如果需要）
            if move_files and category != 'failed':
                success = self.move_image_to_category(image_path, category)
                if success:
                    logger.info(f"  📂 已移动到: {category}")
                else:
                    logger.warning(f"  ⚠️  移动失败")
        
        total_time = time.time() - start_time
        
        # 显示处理总结
        logger.info("="*80)
        logger.info("🎉 图片评分处理完成！")
        logger.info(f"✅ 高分图片(>5分): {stats['high_score']} 张")
        logger.info(f"⚠️  低分图片(1-5分): {stats['low_score']} 张")
        logger.info(f"❌ 零分图片(0分): {stats['zero_score']} 张")
        logger.info(f"❌ 处理失败: {stats['failed']} 张")
        logger.info(f"⏰ 总耗时: {total_time:.1f}秒")
        
        if len(image_files) > 0:
            avg_time = total_time / len(image_files)
            logger.info(f"📊 平均处理时间: {avg_time:.2f}秒/张")
        
        # 显示分数分布
        scores = [r.get('total_score', 0) for r in results if 'total_score' in r]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            logger.info(f"📊 分数统计: 平均{avg_score:.1f}分, 最高{max_score}分, 最低{min_score}分")
            
            # 显示符合要求的比例
            success_rate = (stats['high_score'] / len(image_files)) * 100
            logger.info(f"📈 符合要求比例: {success_rate:.1f}%")
        
        # 保存详细结果
        try:
            import json
            analysis_dir = self.config.get_output_dirs()['analysis']
            
            summary = {
                'total_processed': len(results),
                'statistics': stats,
                'processing_time': total_time,
                'config': {
                    'score_per_face': self.config.SCORE_PER_CLEAR_FRONTAL_FACE,
                    'score_per_plate': self.config.SCORE_PER_CLEAR_PLATE,
                    'score_per_text': self.config.SCORE_PER_TEXT,
                    'required_score': self.config.REQUIRED_TOTAL_SCORE,
                },
                'timestamp': time.time()
            }
            
            # 保存摘要
            summary_file = os.path.join(analysis_dir, "scoring_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # 保存详细结果
            results_file = os.path.join(analysis_dir, "scoring_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 分析结果已保存: {results_file}")
            
        except Exception as e:
            logger.warning(f"保存分析结果失败: {e}")
        
        logger.info("="*80)

def main():
    """主函数"""
    try:
        config = ScoringConfig()
        
        # 检查模型文件
        if not os.path.exists(config.LICENSE_PLATE_MODEL_PATH):
            logger.error(f"❌ 车牌模型不存在: {config.LICENSE_PLATE_MODEL_PATH}")
            return
        
        # 创建评分器
        scorer = StableImageScorer(config)
        
        # 处理图片 - 可以选择测试模式或完整处理
        print("选择处理模式:")
        print("1. 测试模式 (处理前10张)")
        print("2. 小批量处理 (处理前50张)")
        print("3. 完整处理 (处理所有图片)")
        
        try:
            choice = input("请输入选择 (1-3): ").strip()
            move_files = input("是否移动文件到分类目录? (y/n): ").strip().lower() == 'y'
            
            if choice == '1':
                scorer.process_images(max_images=10, move_files=move_files)
            elif choice == '2':
                scorer.process_images(max_images=50, move_files=move_files)
            elif choice == '3':
                scorer.process_images(move_files=move_files)
            else:
                logger.info("默认使用测试模式")
                scorer.process_images(max_images=10, move_files=False)
                
        except KeyboardInterrupt:
            logger.info("⚡ 用户中断")
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断")
    except Exception as e:
        logger.error(f"❌ 程序错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
