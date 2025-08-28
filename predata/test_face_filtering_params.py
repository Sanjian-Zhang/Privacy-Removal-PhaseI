#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同参数组合筛选出最好的4张正脸
"""

import cv2
import numpy as np
import os
import logging
from retinaface import RetinaFace

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_face_quality_score(face_data, img_width, img_height, img_area):
    """计算人脸质量得分"""
    confidence = face_data.get('score', 0.0)
    facial_area = face_data['facial_area']
    landmarks = face_data.get('landmarks', {})
    
    x1, y1, x2, y2 = facial_area
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height
    min_size = min(face_width, face_height)
    
    # 基础得分：置信度 (0-1)
    score = confidence
    
    # 尺寸得分：更大的脸部得分更高
    size_score = min(min_size / 150.0, 1.0)  # 150像素为满分
    score += size_score * 0.3
    
    # 面积比例得分
    face_size_ratio = face_area / img_area
    area_score = min(face_size_ratio / 0.005, 1.0)  # 0.5%面积为满分
    score += area_score * 0.2
    
    # 位置得分：偏向中心的脸部
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    relative_x = face_center_x / img_width
    relative_y = face_center_y / img_height
    
    # 距离图片中心的距离
    center_distance = np.sqrt((relative_x - 0.5)**2 + (relative_y - 0.5)**2)
    position_score = max(0, 1 - center_distance * 2)  # 中心得分最高
    score += position_score * 0.1
    
    # 角度得分：正脸得分更高
    angle_score = 0
    if landmarks:
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width >= 10:
                horizontal_offset = nose[0] - eye_center[0]
                normalized_offset = horizontal_offset / eye_width
                yaw_angle = abs(normalized_offset) * 60.0
                
                # 角度越小得分越高
                angle_score = max(0, 1 - yaw_angle / 25.0)
        except:
            pass
    
    score += angle_score * 0.2
    
    return score, {
        'confidence': confidence,
        'min_size': min_size,
        'face_area': face_area,
        'face_size_ratio': face_size_ratio,
        'center_distance': center_distance,
        'quality_score': score
    }

def test_filtering_combinations(image_path):
    """测试不同的过滤参数组合"""
    
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
    
    # 检测人脸
    logger.info("开始人脸检测...")
    detections = RetinaFace.detect_faces(image_path)
    
    if not isinstance(detections, dict) or len(detections) == 0:
        logger.info("未检测到任何人脸")
        return
    
    logger.info(f"检测到 {len(detections)} 个人脸")
    
    # 计算每个人脸的质量得分
    face_scores = []
    for i, (face_key, face_data) in enumerate(detections.items()):
        score, details = calculate_face_quality_score(face_data, img_width, img_height, img_area)
        face_scores.append({
            'index': i + 1,
            'face_key': face_key,
            'face_data': face_data,
            'score': score,
            'details': details
        })
    
    # 按质量得分排序
    face_scores.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info("\n" + "="*80)
    logger.info("所有人脸按质量得分排序:")
    logger.info("="*80)
    
    for i, face_info in enumerate(face_scores):
        details = face_info['details']
        facial_area = face_info['face_data']['facial_area']
        x1, y1, x2, y2 = facial_area
        
        logger.info(f"排名 #{i+1} (原序号 #{face_info['index']}):")
        logger.info(f"  质量得分: {details['quality_score']:.3f}")
        logger.info(f"  置信度: {details['confidence']:.3f}")
        logger.info(f"  最小边长: {details['min_size']:.0f} 像素")
        logger.info(f"  面积: {details['face_area']:.0f} 像素")
        logger.info(f"  面积比例: {details['face_size_ratio']:.4f} ({details['face_size_ratio']*100:.2f}%)")
        logger.info(f"  距离中心: {details['center_distance']:.3f}")
        logger.info(f"  边界框: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
        logger.info("-" * 40)
    
    # 推荐筛选出前4名的参数组合
    logger.info("\n" + "="*80)
    logger.info("推荐的参数组合来筛选出最好的4张正脸:")
    logger.info("="*80)
    
    if len(face_scores) >= 4:
        # 分析前4名的特征
        top4_faces = face_scores[:4]
        
        min_confidence = min([f['details']['confidence'] for f in top4_faces])
        min_size = min([f['details']['min_size'] for f in top4_faces])
        min_area = min([f['details']['face_area'] for f in top4_faces])
        min_ratio = min([f['details']['face_size_ratio'] for f in top4_faces])
        
        logger.info("方案1 - 保守筛选（确保包含前4名）:")
        logger.info(f"  MIN_FACE_CONFIDENCE_RETINA = {min_confidence - 0.05:.2f}")
        logger.info(f"  MIN_FACE_SIZE = {int(min_size * 0.9)}")
        logger.info(f"  MIN_FACE_AREA = {int(min_area * 0.8)}")
        logger.info(f"  MIN_FACE_SIZE_RATIO = {min_ratio * 0.8:.4f}")
        logger.info(f"  MAX_DISTANCE_THRESHOLD = 0.15")  # 较低的阈值
        logger.info(f"  YAW_ANGLE_THRESHOLD = 25.0")
        
        # 基于第4名和第5名之间的差异
        if len(face_scores) > 4:
            face4_score = face_scores[3]['details']['quality_score']
            face5_score = face_scores[4]['details']['quality_score']
            score_gap = face4_score - face5_score
            
            if score_gap > 0.1:  # 有明显差异
                threshold_confidence = (face_scores[3]['details']['confidence'] + face_scores[4]['details']['confidence']) / 2
                threshold_size = (face_scores[3]['details']['min_size'] + face_scores[4]['details']['min_size']) / 2
                threshold_area = (face_scores[3]['details']['face_area'] + face_scores[4]['details']['face_area']) / 2
                threshold_ratio = (face_scores[3]['details']['face_size_ratio'] + face_scores[4]['details']['face_size_ratio']) / 2
                
                logger.info("\n方案2 - 精确筛选（基于第4名和第5名的分界）:")
                logger.info(f"  MIN_FACE_CONFIDENCE_RETINA = {threshold_confidence:.3f}")
                logger.info(f"  MIN_FACE_SIZE = {int(threshold_size)}")
                logger.info(f"  MIN_FACE_AREA = {int(threshold_area)}")
                logger.info(f"  MIN_FACE_SIZE_RATIO = {threshold_ratio:.4f}")
                logger.info(f"  MAX_DISTANCE_THRESHOLD = 0.20")
                logger.info(f"  YAW_ANGLE_THRESHOLD = 25.0")
        
        # 基于质量的筛选
        avg_top4_score = np.mean([f['details']['quality_score'] for f in top4_faces])
        logger.info("\n方案3 - 基于质量得分筛选:")
        logger.info(f"  使用质量得分阈值: {avg_top4_score * 0.9:.3f}")
        logger.info(f"  或者直接选择得分最高的4个人脸")
        
    else:
        logger.info(f"检测到的人脸数量({len(face_scores)})少于4个，建议降低阈值包含更多人脸")
    
    # 生成具体的配置代码
    logger.info("\n" + "="*80)
    logger.info("具体的代码配置建议:")
    logger.info("="*80)
    
    if len(face_scores) >= 4:
        top4_faces = face_scores[:4]
        min_confidence = min([f['details']['confidence'] for f in top4_faces])
        min_size = min([f['details']['min_size'] for f in top4_faces])
        min_area = min([f['details']['face_area'] for f in top4_faces])
        min_ratio = min([f['details']['face_size_ratio'] for f in top4_faces])
        
        logger.info("# 在您的人脸分类器文件中修改以下参数:")
        logger.info("")
        logger.info("# 方案1 - 保守配置")
        logger.info(f"MIN_FACE_CONFIDENCE_RETINA = {max(0.5, min_confidence - 0.05):.2f}")
        logger.info(f"MIN_FACE_SIZE = {max(30, int(min_size * 0.9))}")
        logger.info(f"MIN_FACE_AREA = {max(900, int(min_area * 0.8))}")
        logger.info(f"MIN_FACE_SIZE_RATIO = {max(0.0005, min_ratio * 0.8):.4f}")
        logger.info("MAX_DISTANCE_THRESHOLD = 0.15")
        logger.info("YAW_ANGLE_THRESHOLD = 25.0")
        logger.info("")
        logger.info("# 或者使用质量得分筛选:")
        logger.info("# 在classify_faces函数中添加质量得分计算，然后选择得分最高的4个")

def main():
    """主函数"""
    # 原图路径
    image_path = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images/0_faces/0_faces/0_faces/video70_frame_000150.jpg"
    
    if not os.path.exists(image_path):
        # 尝试模糊图路径
        image_path = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images_blurred/0_faces/0_faces/0_faces/video70_frame_000150.jpg"
    
    test_filtering_combinations(image_path)

if __name__ == "__main__":
    main()
