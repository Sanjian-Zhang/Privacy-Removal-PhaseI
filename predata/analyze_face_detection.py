#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取指定图片并分析脸框大小，用于调整检测阈值
"""

import cv2
import numpy as np
import os
import logging
from retinaface import RetinaFace

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_face_detection(image_path: str):
    """分析图片中的人脸检测结果"""
    
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        return
    
    logger.info(f"正在分析图片: {image_path}")
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        logger.error("无法读取图片")
        return
    
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    
    logger.info(f"图片尺寸: {img_width} x {img_height}")
    logger.info(f"图片面积: {img_area:,} 像素")
    
    # 使用RetinaFace检测人脸
    logger.info("开始人脸检测...")
    detections = RetinaFace.detect_faces(image_path)
    
    if not isinstance(detections, dict) or len(detections) == 0:
        logger.info("未检测到任何人脸")
        return
    
    logger.info(f"检测到 {len(detections)} 个人脸")
    logger.info("="*60)
    
    # 分析每个检测到的人脸
    for i, (face_key, face_data) in enumerate(detections.items(), 1):
        logger.info(f"人脸 #{i} 分析:")
        
        # 基本信息
        confidence = face_data.get('score', 0.0)
        facial_area = face_data['facial_area']
        landmarks = face_data.get('landmarks', {})
        
        x1, y1, x2, y2 = facial_area
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        
        logger.info(f"  置信度: {confidence:.3f}")
        logger.info(f"  边界框: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
        logger.info(f"  尺寸: {face_width:.0f} x {face_height:.0f}")
        logger.info(f"  面积: {face_area:.0f} 像素")
        
        # 相对于图片的比例
        face_size_ratio = face_area / img_area
        logger.info(f"  面积占比: {face_size_ratio:.4f} ({face_size_ratio*100:.2f}%)")
        
        # 最小边长
        min_size = min(face_width, face_height)
        logger.info(f"  最小边长: {min_size:.0f} 像素")
        
        # 距离比例
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        face_diagonal = np.sqrt(face_width**2 + face_height**2)
        distance_ratio = face_diagonal / img_diagonal
        logger.info(f"  对角线比例: {distance_ratio:.4f}")
        
        # 位置分析
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        relative_x = face_center_x / img_width
        relative_y = face_center_y / img_height
        logger.info(f"  中心位置: ({face_center_x:.0f}, {face_center_y:.0f})")
        logger.info(f"  相对位置: ({relative_x:.3f}, {relative_y:.3f})")
        
        # 边缘检测
        edge_threshold = 0.1
        is_near_edge = (relative_x < edge_threshold or relative_x > (1 - edge_threshold) or
                       relative_y < edge_threshold or relative_y > (1 - edge_threshold))
        logger.info(f"  近边缘: {is_near_edge}")
        
        # yaw角度计算
        if landmarks:
            try:
                left_eye = np.array(landmarks['left_eye'])
                right_eye = np.array(landmarks['right_eye'])
                nose = np.array(landmarks['nose'])
                
                # 计算眼睛中心点
                eye_center = (left_eye + right_eye) / 2
                eye_vector = right_eye - left_eye
                eye_width = np.linalg.norm(eye_vector)
                
                if eye_width >= 10:
                    # 计算鼻子相对于眼睛中心的水平偏移
                    horizontal_offset = nose[0] - eye_center[0]
                    normalized_offset = horizontal_offset / eye_width
                    yaw_angle = abs(normalized_offset) * 60.0
                    
                    logger.info(f"  眼睛距离: {eye_width:.1f} 像素")
                    logger.info(f"  水平偏移: {horizontal_offset:.1f}")
                    logger.info(f"  yaw角度: {yaw_angle:.1f}°")
                    logger.info(f"  是否正脸: {yaw_angle <= 25.0}")
                else:
                    logger.info(f"  眼睛距离太小: {eye_width:.1f} 像素 (可能是侧脸)")
                    
            except Exception as e:
                logger.info(f"  角度计算失败: {e}")
        
        logger.info(f"  当前阈值检查:")
        
        # 当前配置的阈值检查
        current_thresholds = {
            'MIN_FACE_CONFIDENCE_RETINA': 0.7,
            'MIN_FACE_SIZE': 80,
            'MIN_FACE_AREA': 6400,
            'MIN_FACE_SIZE_RATIO': 0.03,
            'MAX_DISTANCE_THRESHOLD': 0.4,
            'YAW_ANGLE_THRESHOLD': 25.0
        }
        
        passes_confidence = confidence >= current_thresholds['MIN_FACE_CONFIDENCE_RETINA']
        passes_size = min_size >= current_thresholds['MIN_FACE_SIZE']
        passes_area = face_area >= current_thresholds['MIN_FACE_AREA']
        passes_ratio = face_size_ratio >= current_thresholds['MIN_FACE_SIZE_RATIO']
        passes_distance = distance_ratio >= current_thresholds['MAX_DISTANCE_THRESHOLD']
        
        logger.info(f"    置信度 >= {current_thresholds['MIN_FACE_CONFIDENCE_RETINA']}: {passes_confidence}")
        logger.info(f"    最小尺寸 >= {current_thresholds['MIN_FACE_SIZE']}: {passes_size}")
        logger.info(f"    面积 >= {current_thresholds['MIN_FACE_AREA']}: {passes_area}")
        logger.info(f"    面积比例 >= {current_thresholds['MIN_FACE_SIZE_RATIO']}: {passes_ratio}")
        logger.info(f"    距离比例 >= {current_thresholds['MAX_DISTANCE_THRESHOLD']}: {passes_distance}")
        
        overall_pass = all([passes_confidence, passes_size, passes_area, passes_ratio, passes_distance])
        logger.info(f"    总体通过: {overall_pass}")
        
        logger.info("-" * 40)
    
    # 建议阈值
    logger.info("建议的阈值调整:")
    logger.info("基于上述分析结果，您可以考虑以下调整:")
    
    # 收集所有人脸的统计信息
    all_confidences = [face_data.get('score', 0.0) for face_data in detections.values()]
    all_areas = []
    all_min_sizes = []
    all_ratios = []
    all_distance_ratios = []
    
    img_diagonal = np.sqrt(img_width**2 + img_height**2)  # 预先计算图片对角线
    
    for face_data in detections.values():
        x1, y1, x2, y2 = face_data['facial_area']
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        min_size = min(face_width, face_height)
        face_size_ratio = face_area / img_area
        face_diagonal = np.sqrt(face_width**2 + face_height**2)
        distance_ratio = face_diagonal / img_diagonal
        
        all_areas.append(face_area)
        all_min_sizes.append(min_size)
        all_ratios.append(face_size_ratio)
        all_distance_ratios.append(distance_ratio)
    
    if all_confidences:
        min_conf = min(all_confidences)
        avg_conf = np.mean(all_confidences)
        logger.info(f"  置信度范围: {min_conf:.3f} - {max(all_confidences):.3f} (平均: {avg_conf:.3f})")
        
    if all_min_sizes:
        min_size_val = min(all_min_sizes)
        avg_size = np.mean(all_min_sizes)
        logger.info(f"  最小尺寸范围: {min_size_val:.0f} - {max(all_min_sizes):.0f} (平均: {avg_size:.0f})")
        
    if all_ratios:
        min_ratio = min(all_ratios)
        avg_ratio = np.mean(all_ratios)
        logger.info(f"  面积比例范围: {min_ratio:.4f} - {max(all_ratios):.4f} (平均: {avg_ratio:.4f})")

def suggest_thresholds():
    """根据分析结果建议新的阈值"""
    logger.info("="*60)
    logger.info("基于这张图片的分析，建议的阈值调整方案:")
    
    suggestions = [
        "1. 如果要包含所有检测到的人脸，可以降低以下阈值:",
        "   - MIN_FACE_SIZE: 从 80 降低到 60 或 50",
        "   - MIN_FACE_AREA: 从 6400 降低到 3600 或 2500", 
        "   - MIN_FACE_SIZE_RATIO: 从 0.03 降低到 0.02 或 0.015",
        "   - MAX_DISTANCE_THRESHOLD: 从 0.4 降低到 0.3 或 0.25",
        "",
        "2. 如果要过滤掉远距离或不清晰的人脸，可以提高阈值:",
        "   - MIN_FACE_SIZE: 从 80 提高到 100 或 120",
        "   - MIN_FACE_CONFIDENCE_RETINA: 从 0.7 提高到 0.8 或 0.85",
        "",
        "3. 角度阈值调整:",
        "   - YAW_ANGLE_THRESHOLD: 保持 25.0 或调整到 20.0 (更严格) 或 30.0 (更宽松)"
    ]
    
    for suggestion in suggestions:
        logger.info(suggestion)

def main():
    """主函数"""
    image_path = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images_blurred/0_faces/0_faces/0_faces/video70_frame_000150.jpg"
    
    # 检查是否为模糊处理后的图片路径
    if "high_score_images_blurred" in image_path:
        # 如果是模糊后的图片，同时分析原图
        original_path = image_path.replace("high_score_images_blurred", "high_score_images")
        if os.path.exists(original_path):
            logger.info("检测到模糊处理后的图片，将同时分析原图")
            logger.info("="*60)
            logger.info("原图分析:")
            analyze_face_detection(original_path)
            logger.info("\n" + "="*60)
            logger.info("模糊图分析:")
    
    analyze_face_detection(image_path)
    suggest_thresholds()

if __name__ == "__main__":
    main()
