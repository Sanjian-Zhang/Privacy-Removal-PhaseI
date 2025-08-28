#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的正脸检测器 - 解决后脑勺误判问题
使用多重验证确保检测到的是真正的正脸而不是后脑勺
"""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
import logging
import os
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    logger.info("✅ RetinaFace库可用")
except ImportError:
    RETINAFACE_AVAILABLE = False
    logger.warning("⚠️ RetinaFace库不可用")

class ImprovedFaceDetector:
    """改进的正脸检测器，专门解决后脑勺误判问题"""
    
    def __init__(self, 
                 min_confidence=0.8,
                 max_yaw_angle=20.0,        # 更严格的yaw角度阈值
                 max_pitch_angle=25.0,      # pitch角度阈值（上下倾斜）
                 max_roll_angle=30.0,       # roll角度阈值（旋转）
                 min_face_size=150,         # 更大的最小人脸尺寸
                 min_area_ratio=0.015,      # 更大的最小面积比例
                 eye_visibility_threshold=0.7,  # 眼部可见性阈值
                 enable_eye_detection=True,      # 启用独立眼部检测
                 enable_profile_rejection=True): # 启用侧脸拒绝
        
        self.min_confidence = min_confidence
        self.max_yaw_angle = max_yaw_angle
        self.max_pitch_angle = max_pitch_angle
        self.max_roll_angle = max_roll_angle
        self.min_face_size = min_face_size
        self.min_area_ratio = min_area_ratio
        self.eye_visibility_threshold = eye_visibility_threshold
        self.enable_eye_detection = enable_eye_detection
        self.enable_profile_rejection = enable_profile_rejection
        
        # 加载Haar级联分类器用于眼部检测
        if self.enable_eye_detection:
            self.eye_cascade = self._load_eye_cascade()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'faces_detected': 0,
            'frontal_faces_found': 0,
            'rejected_by_yaw': 0,
            'rejected_by_pitch': 0,
            'rejected_by_roll': 0,
            'rejected_by_eyes': 0,
            'rejected_by_profile': 0,
            'rejected_by_size': 0
        }
    
    def _load_eye_cascade(self):
        """加载眼部检测级联分类器"""
        try:
            # 尝试多个可能的路径
            eye_cascade_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml',
                'haarcascade_eye.xml',
                './haarcascade_eye.xml'
            ]
            
            # 尝试获取OpenCV数据路径
            try:
                import cv2.data
                eye_cascade_paths.append(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml'))
            except:
                pass
            
            for path in eye_cascade_paths:
                if os.path.exists(path):
                    logger.info(f"✅ 加载眼部检测器: {path}")
                    return cv2.CascadeClassifier(path)
            
            logger.warning("⚠️ 未找到眼部检测器，将禁用眼部验证")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 眼部检测器加载失败: {e}")
            return None
    
    def calculate_advanced_pose_angles(self, landmarks: Dict) -> Tuple[float, float, float]:
        """
        计算更精确的人脸姿态角度
        
        Returns:
            tuple: (yaw, pitch, roll) 角度（度）
        """
        try:
            # 获取关键点
            left_eye = np.array(landmarks.get('left_eye', [0, 0]))
            right_eye = np.array(landmarks.get('right_eye', [0, 0]))
            nose = np.array(landmarks.get('nose', [0, 0]))
            left_mouth = np.array(landmarks.get('left_mouth_corner', [0, 0]))
            right_mouth = np.array(landmarks.get('right_mouth_corner', [0, 0]))
            
            # 如果缺少关键点，返回大角度
            if np.allclose(left_eye, [0, 0]) or np.allclose(right_eye, [0, 0]) or np.allclose(nose, [0, 0]):
                return 90.0, 90.0, 90.0
            
            # 1. 计算 Yaw 角度（左右转头）
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width < 10:
                return 90.0, 90.0, 90.0
            
            # 改进的yaw计算：基于鼻子相对于眼部中心的偏移
            nose_to_eye_center = nose - eye_center
            horizontal_offset = nose_to_eye_center[0]
            
            # 标准化偏移量
            normalized_offset = horizontal_offset / eye_width
            yaw_angle = abs(normalized_offset) * 45.0  # 降低系数，更保守
            
            # 2. 计算 Pitch 角度（上下点头）
            if not np.allclose(left_mouth, [0, 0]) and not np.allclose(right_mouth, [0, 0]):
                mouth_center = (left_mouth + right_mouth) / 2
                eye_mouth_vector = mouth_center - eye_center
                vertical_distance = abs(eye_mouth_vector[1])
                
                # 正常情况下，眼部到嘴部的垂直距离应该在合理范围内
                expected_eye_mouth_ratio = 0.6  # 经验值
                actual_ratio = vertical_distance / eye_width if eye_width > 0 else 1.0
                
                pitch_deviation = abs(actual_ratio - expected_eye_mouth_ratio) / expected_eye_mouth_ratio
                pitch_angle = pitch_deviation * 30.0
            else:
                pitch_angle = 0.0
            
            # 3. 计算 Roll 角度（头部倾斜）
            if eye_width > 0:
                eye_slope = (right_eye[1] - left_eye[1]) / eye_width
                roll_angle = abs(math.atan(eye_slope) * 180 / math.pi)
            else:
                roll_angle = 90.0
            
            return min(yaw_angle, 90.0), min(pitch_angle, 90.0), min(roll_angle, 90.0)
            
        except Exception as e:
            logger.debug(f"姿态角度计算失败: {e}")
            return 90.0, 90.0, 90.0
    
    def analyze_facial_symmetry(self, landmarks: Dict) -> float:
        """
        分析面部对称性，正脸应该更对称
        
        Returns:
            float: 对称性分数 (0-1，1表示完全对称)
        """
        try:
            left_eye = np.array(landmarks.get('left_eye', [0, 0]))
            right_eye = np.array(landmarks.get('right_eye', [0, 0]))
            nose = np.array(landmarks.get('nose', [0, 0]))
            left_mouth = np.array(landmarks.get('left_mouth_corner', [0, 0]))
            right_mouth = np.array(landmarks.get('right_mouth_corner', [0, 0]))
            
            if any(np.allclose(point, [0, 0]) for point in [left_eye, right_eye, nose, left_mouth, right_mouth]):
                return 0.0
            
            # 计算面部中轴线（通过鼻子，垂直于双眼连线）
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            
            # 鼻子应该在眼部中心线上（对于正脸）
            nose_deviation = abs(np.dot(nose - eye_center, eye_vector)) / np.linalg.norm(eye_vector)
            
            # 嘴部中心也应该在中轴线上
            mouth_center = (left_mouth + right_mouth) / 2
            mouth_deviation = abs(np.dot(mouth_center - eye_center, eye_vector)) / np.linalg.norm(eye_vector)
            
            # 计算对称性分数
            max_deviation = max(nose_deviation, mouth_deviation)
            symmetry_score = max(0, 1 - max_deviation / 20)  # 20像素作为基准
            
            return symmetry_score
            
        except Exception as e:
            logger.debug(f"对称性分析失败: {e}")
            return 0.0
    
    def detect_eyes_independently(self, img: np.ndarray, face_region: Tuple[int, int, int, int]) -> bool:
        """
        独立检测眼部，确认是否能看到双眼
        
        Args:
            img: 输入图像
            face_region: 人脸区域 (x1, y1, x2, y2)
            
        Returns:
            bool: 是否检测到足够的眼部特征
        """
        if self.eye_cascade is None:
            return True  # 如果没有眼部检测器，跳过此检查
        
        try:
            x1, y1, x2, y2 = face_region
            face_roi = img[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return False
            
            # 转换为灰度图
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # 检测眼部
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10),
                maxSize=(100, 100)
            )
            
            # 正脸应该能检测到至少1-2只眼睛
            eye_count = len(eyes)
            
            # 分析眼部分布
            if eye_count >= 2:
                # 检查眼部是否在合理位置（上半部分）
                face_height = y2 - y1
                eyes_in_upper_half = sum(1 for (ex, ey, ew, eh) in eyes if ey < face_height * 0.6)
                
                return eyes_in_upper_half >= 2
            elif eye_count == 1:
                # 单眼检测可能是侧脸，但也可能是遮挡，需要更多验证
                return True
            else:
                # 没有检测到眼部，可能是后脑勺
                return False
                
        except Exception as e:
            logger.debug(f"独立眼部检测失败: {e}")
            return True  # 检测失败时保守处理
    
    def analyze_facial_features_distribution(self, landmarks: Dict, face_area: Tuple[int, int, int, int]) -> bool:
        """
        分析面部特征分布，判断是否符合正脸特征
        
        Args:
            landmarks: 面部关键点
            face_area: 人脸区域
            
        Returns:
            bool: 是否符合正脸特征分布
        """
        try:
            x1, y1, x2, y2 = face_area
            face_width = x2 - x1
            face_height = y2 - y1
            
            left_eye = np.array(landmarks.get('left_eye', [0, 0]))
            right_eye = np.array(landmarks.get('right_eye', [0, 0]))
            nose = np.array(landmarks.get('nose', [0, 0]))
            left_mouth = np.array(landmarks.get('left_mouth_corner', [0, 0]))
            right_mouth = np.array(landmarks.get('right_mouth_corner', [0, 0]))
            
            # 检查关键点是否都在面部区域内
            for point_name, point in [('left_eye', left_eye), ('right_eye', right_eye), 
                                     ('nose', nose), ('left_mouth', left_mouth), ('right_mouth', right_mouth)]:
                if np.allclose(point, [0, 0]):
                    continue
                
                # 转换为相对坐标
                rel_x = (point[0] - x1) / face_width
                rel_y = (point[1] - y1) / face_height
                
                # 检查点是否在合理范围内
                if not (0.1 <= rel_x <= 0.9 and 0.1 <= rel_y <= 0.9):
                    logger.debug(f"关键点 {point_name} 超出合理范围: ({rel_x:.2f}, {rel_y:.2f})")
                    return False
            
            # 检查眼部间距是否合理
            if not np.allclose(left_eye, [0, 0]) and not np.allclose(right_eye, [0, 0]):
                eye_distance = np.linalg.norm(right_eye - left_eye)
                eye_distance_ratio = eye_distance / face_width
                
                # 正脸的眼间距通常占面部宽度的25%-45%
                if not (0.20 <= eye_distance_ratio <= 0.50):
                    logger.debug(f"眼间距比例异常: {eye_distance_ratio:.2f}")
                    return False
            
            # 检查五官垂直分布
            if not any(np.allclose(point, [0, 0]) for point in [left_eye, right_eye, nose, left_mouth, right_mouth]):
                eye_center = (left_eye + right_eye) / 2
                mouth_center = (left_mouth + right_mouth) / 2
                
                # 相对位置检查
                eye_y_ratio = (eye_center[1] - y1) / face_height
                nose_y_ratio = (nose[1] - y1) / face_height
                mouth_y_ratio = (mouth_center[1] - y1) / face_height
                
                # 正脸的典型比例：眼部20%-40%，鼻子40%-65%，嘴部65%-85%
                if not (0.15 <= eye_y_ratio <= 0.45):
                    logger.debug(f"眼部垂直位置异常: {eye_y_ratio:.2f}")
                    return False
                
                if not (0.35 <= nose_y_ratio <= 0.70):
                    logger.debug(f"鼻子垂直位置异常: {nose_y_ratio:.2f}")
                    return False
                
                if not (0.60 <= mouth_y_ratio <= 0.90):
                    logger.debug(f"嘴部垂直位置异常: {mouth_y_ratio:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"特征分布分析失败: {e}")
            return True  # 分析失败时保守处理
    
    def is_frontal_face(self, img: np.ndarray, face_data: Dict) -> Tuple[bool, Dict]:
        """
        综合判断是否为正脸
        
        Args:
            img: 输入图像
            face_data: RetinaFace检测结果
            
        Returns:
            tuple: (是否为正脸, 详细分析结果)
        """
        try:
            # 基本信息提取
            confidence = face_data.get('score', 0.0)
            facial_area = face_data['facial_area']
            landmarks = face_data.get('landmarks', {})
            
            x1, y1, x2, y2 = facial_area
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            area_ratio = face_area / img_area
            
            analysis_result = {
                'confidence': confidence,
                'face_size': (face_width, face_height),
                'area_ratio': area_ratio,
                'rejection_reasons': []
            }
            
            # 1. 置信度检查
            if confidence < self.min_confidence:
                analysis_result['rejection_reasons'].append(f"置信度过低: {confidence:.3f} < {self.min_confidence}")
                return False, analysis_result
            
            # 2. 尺寸检查
            if min(face_width, face_height) < self.min_face_size:
                analysis_result['rejection_reasons'].append(f"尺寸过小: {min(face_width, face_height)} < {self.min_face_size}")
                self.stats['rejected_by_size'] += 1
                return False, analysis_result
            
            if area_ratio < self.min_area_ratio:
                analysis_result['rejection_reasons'].append(f"面积比例过小: {area_ratio:.4f} < {self.min_area_ratio}")
                self.stats['rejected_by_size'] += 1
                return False, analysis_result
            
            # 3. 关键点检查
            if not landmarks:
                analysis_result['rejection_reasons'].append("缺少面部关键点")
                return False, analysis_result
            
            # 4. 姿态角度分析
            yaw, pitch, roll = self.calculate_advanced_pose_angles(landmarks)
            analysis_result.update({
                'yaw_angle': yaw,
                'pitch_angle': pitch,
                'roll_angle': roll
            })
            
            if yaw > self.max_yaw_angle:
                analysis_result['rejection_reasons'].append(f"Yaw角度过大: {yaw:.1f}° > {self.max_yaw_angle}°")
                self.stats['rejected_by_yaw'] += 1
                return False, analysis_result
            
            if pitch > self.max_pitch_angle:
                analysis_result['rejection_reasons'].append(f"Pitch角度过大: {pitch:.1f}° > {self.max_pitch_angle}°")
                self.stats['rejected_by_pitch'] += 1
                return False, analysis_result
            
            if roll > self.max_roll_angle:
                analysis_result['rejection_reasons'].append(f"Roll角度过大: {roll:.1f}° > {self.max_roll_angle}°")
                self.stats['rejected_by_roll'] += 1
                return False, analysis_result
            
            # 5. 面部对称性检查
            symmetry_score = self.analyze_facial_symmetry(landmarks)
            analysis_result['symmetry_score'] = symmetry_score
            
            if symmetry_score < 0.3:  # 对称性阈值
                analysis_result['rejection_reasons'].append(f"对称性过低: {symmetry_score:.3f} < 0.3")
                self.stats['rejected_by_profile'] += 1
                return False, analysis_result
            
            # 6. 独立眼部检测
            if self.enable_eye_detection:
                eyes_detected = self.detect_eyes_independently(img, facial_area)
                analysis_result['eyes_detected'] = eyes_detected
                
                if not eyes_detected:
                    analysis_result['rejection_reasons'].append("未检测到足够的眼部特征")
                    self.stats['rejected_by_eyes'] += 1
                    return False, analysis_result
            
            # 7. 特征分布检查
            if self.enable_profile_rejection:
                features_valid = self.analyze_facial_features_distribution(landmarks, facial_area)
                analysis_result['features_distribution_valid'] = features_valid
                
                if not features_valid:
                    analysis_result['rejection_reasons'].append("面部特征分布不符合正脸模式")
                    self.stats['rejected_by_profile'] += 1
                    return False, analysis_result
            
            # 8. 综合评分
            composite_score = (
                (1 - yaw / 90.0) * 0.3 +           # yaw角度权重
                (1 - pitch / 90.0) * 0.2 +         # pitch角度权重
                (1 - roll / 90.0) * 0.2 +          # roll角度权重
                symmetry_score * 0.2 +              # 对称性权重
                min(area_ratio / 0.1, 1.0) * 0.1   # 大小权重
            )
            
            analysis_result['composite_score'] = composite_score
            
            # 综合评分阈值
            if composite_score < 0.6:
                analysis_result['rejection_reasons'].append(f"综合评分过低: {composite_score:.3f} < 0.6")
                return False, analysis_result
            
            # 通过所有检查
            self.stats['frontal_faces_found'] += 1
            return True, analysis_result
            
        except Exception as e:
            logger.error(f"正脸判断失败: {e}")
            analysis_result = {
                'confidence': 0.0,
                'face_size': (0, 0),
                'area_ratio': 0.0,
                'rejection_reasons': [f"分析异常: {str(e)}"]
            }
            return False, analysis_result
    
    def detect_frontal_faces(self, image_path: str, return_details: bool = False) -> Tuple[int, List]:
        """
        检测图片中的正脸数量
        
        Args:
            image_path: 图片路径
            return_details: 是否返回详细分析结果
            
        Returns:
            tuple: (正脸数量, 详细结果列表)
        """
        self.stats['total_processed'] += 1
        
        if not RETINAFACE_AVAILABLE:
            logger.error("RetinaFace不可用，无法进行检测")
            return 0, []
        
        try:
            # 使用RetinaFace检测
            from retinaface import RetinaFace
            detections = RetinaFace.detect_faces(image_path)
            
            if not isinstance(detections, dict) or len(detections) == 0:
                return 0, []
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图像: {image_path}")
                return 0, []
            
            self.stats['faces_detected'] += len(detections)
            
            frontal_count = 0
            detailed_results = []
            
            for face_key, face_data in detections.items():
                is_frontal, analysis = self.is_frontal_face(img, face_data)
                
                if return_details:
                    detailed_results.append({
                        'face_key': face_key,
                        'is_frontal': is_frontal,
                        'analysis': analysis
                    })
                
                if is_frontal:
                    frontal_count += 1
            
            return frontal_count, detailed_results if return_details else []
            
        except Exception as e:
            logger.error(f"检测失败 {image_path}: {e}")
            return 0, []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_faces = self.stats['faces_detected']
        total_processed = self.stats['total_processed']
        
        # 创建一个新的字典来避免类型问题
        stats = {}
        
        # 复制基础统计信息
        for key, value in self.stats.items():
            stats[key] = value
        
        if total_faces > 0:
            stats['frontal_rate'] = self.stats['frontal_faces_found'] / total_faces
            stats['rejection_breakdown'] = {
                'yaw_percentage': self.stats['rejected_by_yaw'] / total_faces * 100,
                'pitch_percentage': self.stats['rejected_by_pitch'] / total_faces * 100,
                'roll_percentage': self.stats['rejected_by_roll'] / total_faces * 100,
                'eyes_percentage': self.stats['rejected_by_eyes'] / total_faces * 100,
                'profile_percentage': self.stats['rejected_by_profile'] / total_faces * 100,
                'size_percentage': self.stats['rejected_by_size'] / total_faces * 100
            }
        
        if total_processed > 0:
            stats['average_faces_per_image'] = total_faces / total_processed
            stats['images_with_frontal_faces'] = sum(1 for _ in range(total_processed) if self.stats['frontal_faces_found'] > 0)
        
        return stats

def test_improved_detector():
    """测试改进的检测器"""
    detector = ImprovedFaceDetector(
        min_confidence=0.85,
        max_yaw_angle=15.0,      # 更严格的yaw角度
        max_pitch_angle=20.0,    # 更严格的pitch角度
        max_roll_angle=25.0,     # 更严格的roll角度
        min_face_size=120,       # 更大的最小尺寸
        min_area_ratio=0.02,     # 更大的最小面积比例
        enable_eye_detection=True,
        enable_profile_rejection=True
    )
    
    # 测试目录
    test_dir = "/home/zhiqics/sanjian/predata/output_frames70"
    
    if not os.path.exists(test_dir):
        logger.error(f"测试目录不存在: {test_dir}")
        return
    
    # 获取测试图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(Path(test_dir).glob(f"*{ext}"))
    
    test_images = test_images[:20]  # 只测试前20张
    
    logger.info(f"开始测试，图片数量: {len(test_images)}")
    
    results = []
    for img_path in test_images:
        frontal_count, details = detector.detect_frontal_faces(str(img_path), return_details=True)
        results.append({
            'image': img_path.name,
            'frontal_count': frontal_count,
            'details': details
        })
        
        logger.info(f"📸 {img_path.name}: 检测到 {frontal_count} 张正脸")
        
        # 显示详细分析
        for detail in details:
            analysis = detail['analysis']
            if detail['is_frontal']:
                logger.info(f"  ✅ 正脸 - 综合评分: {analysis.get('composite_score', 0):.3f}")
            else:
                reasons = ', '.join(analysis.get('rejection_reasons', []))
                logger.info(f"  ❌ 拒绝 - 原因: {reasons}")
    
    # 显示统计信息
    stats = detector.get_statistics()
    logger.info("\n📊 检测统计:")
    logger.info(f"  处理图片: {stats['total_processed']}")
    logger.info(f"  检测到人脸: {stats['faces_detected']}")
    logger.info(f"  正脸数量: {stats['frontal_faces_found']}")
    
    if 'frontal_rate' in stats:
        logger.info(f"  正脸比例: {stats['frontal_rate']:.1%}")
    
    if 'rejection_breakdown' in stats:
        breakdown = stats['rejection_breakdown']
        logger.info("  拒绝原因分布:")
        logger.info(f"    Yaw角度: {breakdown['yaw_percentage']:.1f}%")
        logger.info(f"    Pitch角度: {breakdown['pitch_percentage']:.1f}%")
        logger.info(f"    Roll角度: {breakdown['roll_percentage']:.1f}%")
        logger.info(f"    眼部检测: {breakdown['eyes_percentage']:.1f}%")
        logger.info(f"    侧脸/特征: {breakdown['profile_percentage']:.1f}%")
        logger.info(f"    尺寸过小: {breakdown['size_percentage']:.1f}%")

if __name__ == "__main__":
    test_improved_detector()
