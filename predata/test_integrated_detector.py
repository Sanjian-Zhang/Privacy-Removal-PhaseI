#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集成后的快速人脸车牌检测器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_face_plate_detector_v2 import FastConfig, FastProcessor, get_image_files
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_integrated_detector():
    """测试集成后的检测器"""
    
    # 创建测试配置
    class TestConfig(FastConfig):
        # 使用较小的测试数据集
        INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames15/two_plates'
        OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/test_output'
        
        # 调整参数以适合测试
        BATCH_SIZE = 8
        ENABLE_SIMILARITY_DETECTION = True
        PSNR_THRESHOLD = 50.0
        ADJACENT_FRAME_THRESHOLD = 8
        MIN_FRAME_DISTANCE = 5
        
        # 降低GPU内存需求
        MAX_GPU_MEMORY_MB = 2048
        MAX_CPU_MEMORY_MB = 1024
    
    config = TestConfig()
    
    logger.info("🧪 开始测试集成后的快速人脸车牌检测器...")
    logger.info(f"📁 测试输入目录: {config.INPUT_DIR}")
    logger.info(f"📁 测试输出目录: {config.OUTPUT_BASE_DIR}")
    
    # 检查输入目录
    if not os.path.exists(config.INPUT_DIR):
        logger.error(f"❌ 测试输入目录不存在: {config.INPUT_DIR}")
        return False
    
    # 检查图片文件
    image_files = get_image_files(config.INPUT_DIR)
    if len(image_files) == 0:
        logger.error("❌ 测试目录中没有找到图片文件")
        return False
    
    logger.info(f"📊 找到 {len(image_files)} 张测试图片")
    
    try:
        # 创建处理器并运行
        processor = FastProcessor(config)
        
        # 检查模型文件
        if not os.path.exists(config.YOLOV8S_MODEL_PATH):
            logger.warning(f"⚠️ YOLOv8s模型不存在，使用默认模型: {config.YOLOV8S_MODEL_PATH}")
            # 可以下载或使用默认模型
            config.YOLOV8S_MODEL_PATH = 'yolov8s.pt'  # 使用ultralytics的默认模型
        
        if not os.path.exists(config.LICENSE_PLATE_MODEL_PATH):
            logger.warning(f"⚠️ 车牌检测模型不存在，跳过车牌检测")
            # 可以临时禁用车牌检测或使用备用模型
        
        processor.run()
        
        logger.info("✅ 集成测试完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_integrated_detector()
    if success:
        print("\n🎉 集成测试成功！")
    else:
        print("\n❌ 集成测试失败！")
        sys.exit(1)
