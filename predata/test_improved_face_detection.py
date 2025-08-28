#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的正脸检测器 - 验证后脑勺过滤效果
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.append('/home/zhiqics/sanjian/predata')

from improved_face_detector import ImprovedFaceDetector
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_face_detection_improvements():
    """测试改进的人脸检测效果"""
    
    # 创建改进的检测器
    detector = ImprovedFaceDetector(
        min_confidence=0.85,          # 更高的置信度
        max_yaw_angle=15.0,           # 更严格的yaw角度
        max_pitch_angle=20.0,         # pitch角度限制
        max_roll_angle=25.0,          # roll角度限制
        min_face_size=140,            # 更大的最小尺寸
        min_area_ratio=0.015,         # 更大的面积比例
        enable_eye_detection=True,     # 启用眼部检测
        enable_profile_rejection=True  # 启用侧脸/后脑勺拒绝
    )
    
    # 测试目录
    test_dirs = [
        "/home/zhiqics/sanjian/predata/output_frames70",
        "/home/zhiqics/sanjian/predata/output_frames15", 
        "/home/zhiqics/sanjian/predata/output_frames16"
    ]
    
    # 找到存在的测试目录
    test_dir = None
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            test_dir = dir_path
            break
    
    if not test_dir:
        logger.error("未找到测试目录")
        return
    
    logger.info(f"🔍 测试目录: {test_dir}")
    
    # 获取测试图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(Path(test_dir).glob(f"*{ext}"))
    
    # 限制测试数量
    test_images = sorted(test_images)[:30]
    
    if not test_images:
        logger.error("未找到测试图片")
        return
    
    logger.info(f"📊 测试图片数量: {len(test_images)}")
    logger.info("="*80)
    
    frontal_faces_total = 0
    images_with_frontal = 0
    
    # 逐一测试
    for i, img_path in enumerate(test_images, 1):
        logger.info(f"\n📸 [{i:2d}/{len(test_images)}] {img_path.name}")
        
        frontal_count, details = detector.detect_frontal_faces(str(img_path), return_details=True)
        
        if frontal_count > 0:
            frontal_faces_total += frontal_count
            images_with_frontal += 1
            logger.info(f"   ✅ 检测到 {frontal_count} 张正脸")
            
            # 显示详细分析
            for j, detail in enumerate(details, 1):
                analysis = detail['analysis']
                if detail['is_frontal']:
                    logger.info(f"      💚 正脸{j}: 评分={analysis.get('composite_score', 0):.3f}, "
                              f"yaw={analysis.get('yaw_angle', 0):.1f}°, "
                              f"对称性={analysis.get('symmetry_score', 0):.3f}")
                else:
                    reasons = analysis.get('rejection_reasons', [])
                    main_reason = reasons[0] if reasons else "未知原因"
                    logger.info(f"      ❌ 拒绝{j}: {main_reason}")
        else:
            logger.info(f"   ❌ 未检测到正脸")
            
            # 显示拒绝原因
            for j, detail in enumerate(details, 1):
                if not detail['is_frontal']:
                    analysis = detail['analysis']
                    reasons = analysis.get('rejection_reasons', [])
                    if reasons:
                        logger.info(f"      🔍 拒绝原因: {reasons[0]}")
    
    # 汇总统计
    logger.info("\n" + "="*80)
    logger.info("📊 测试结果统计:")
    
    stats = detector.get_statistics()
    
    logger.info(f"   📸 测试图片总数: {stats['total_processed']}")
    logger.info(f"   👤 检测到人脸总数: {stats['faces_detected']}")
    logger.info(f"   ✅ 正脸总数: {stats['frontal_faces_found']}")
    logger.info(f"   📈 有正脸的图片: {images_with_frontal}")
    logger.info(f"   📊 正脸检出率: {images_with_frontal/len(test_images)*100:.1f}%")
    
    if stats['faces_detected'] > 0:
        logger.info(f"   🎯 正脸准确率: {stats['frontal_faces_found']/stats['faces_detected']*100:.1f}%")
    
    # 拒绝原因分析
    if 'rejection_breakdown' in stats:
        logger.info(f"\n🔍 拒绝原因分布:")
        breakdown = stats['rejection_breakdown']
        logger.info(f"   📐 Yaw角度过大: {breakdown['yaw_percentage']:.1f}%")
        logger.info(f"   📐 Pitch角度过大: {breakdown['pitch_percentage']:.1f}%")
        logger.info(f"   📐 Roll角度过大: {breakdown['roll_percentage']:.1f}%")
        logger.info(f"   👁️  眼部检测失败: {breakdown['eyes_percentage']:.1f}%")
        logger.info(f"   👤 侧脸/后脑勺: {breakdown['profile_percentage']:.1f}%")
        logger.info(f"   📏 尺寸过小: {breakdown['size_percentage']:.1f}%")
    
    logger.info("\n✨ 改进效果:")
    logger.info("   1. 更严格的yaw角度阈值 (15°)")
    logger.info("   2. 增加pitch和roll角度检查")
    logger.info("   3. 面部对称性验证")
    logger.info("   4. 眼部独立检测验证")
    logger.info("   5. 特征分布合理性检查")
    logger.info("   6. 更大的最小人脸尺寸要求")

def compare_before_after():
    """对比改进前后的效果"""
    logger.info("\n🔄 对比改进前后效果...")
    
    # 可以在这里添加与原始检测器的对比逻辑
    # 比如运行原始的2-fast_face_plate_detector_v2.py
    # 然后对比结果
    
    logger.info("💡 建议:")
    logger.info("   1. 如果仍有后脑勺误判，可进一步降低yaw_angle阈值")
    logger.info("   2. 如果正脸检出率过低，可适当放宽某些条件")
    logger.info("   3. 可以考虑增加手动验证步骤")

if __name__ == "__main__":
    logger.info("🧪 开始测试改进的正脸检测器")
    logger.info("="*80)
    
    test_face_detection_improvements()
    compare_before_after()
    
    logger.info("\n🎉 测试完成!")
    logger.info("💡 如需调整参数，请修改 improved_face_detector.py 中的阈值设置")
