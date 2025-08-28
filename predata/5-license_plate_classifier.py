#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车牌检测分类器
使用YOLOv8模型检测图片中的车牌并按数量分类
作者: GitHub Copilot
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    print("警告: ultralytics库未安装，请运行: pip install ultralytics")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('license_plate_classifier.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LicensePlateClassifier:
    """车牌分类器"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.3):
        """
        初始化车牌分类器
        
        Args:
            model_path: YOLO模型路径
            confidence_threshold: 置信度阈值
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics库未安装")
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.stats = {
            'total_processed': 0,
            'total_plates_detected': 0,
            'classification_counts': {}
        }
        
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            logger.info(f"🔄 加载车牌检测模型: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            if not YOLO_AVAILABLE or YOLO is None:
                raise ImportError("ultralytics库未安装或导入失败")
            
            self.model = YOLO(self.model_path)
            logger.info("✅ 车牌检测模型加载成功")
            
            # 测试GPU可用性
            import torch
            if torch.cuda.is_available():
                logger.info(f"🚀 检测到GPU: {torch.cuda.get_device_name()}")
                self.model.to('cuda')
            else:
                logger.info("💻 使用CPU进行推理")
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def detect_license_plates(self, image_path: str) -> Tuple[int, List[Dict]]:
        """
        检测图片中的车牌
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (车牌数量, 检测结果列表)
        """
        try:
            # 读取图片
            if not os.path.exists(image_path):
                logger.warning(f"图片不存在: {image_path}")
                return 0, []
            
            if self.model is None:
                logger.error("模型未加载")
                return 0, []
            
            # 使用YOLO进行检测
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            
            # 解析检测结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2, conf, class_id = box
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        })
            
            plate_count = len(detections)
            self.stats['total_plates_detected'] += plate_count
            
            return plate_count, detections
            
        except Exception as e:
            logger.error(f"检测失败 {image_path}: {e}")
            return 0, []
    
    def classify_by_plate_count(self, plate_count: int) -> str:
        """
        根据车牌数量返回分类名称
        
        Args:
            plate_count: 车牌数量
            
        Returns:
            str: 分类文件夹名称
        """
        if plate_count == 0:
            return "no_plates"
        elif plate_count == 1:
            return "one_plate"
        elif plate_count == 2:
            return "two_plates"
        elif plate_count == 3:
            return "three_plates"
        else:
            return "multiple_plates"
    
    def process_images(self, input_dir: str, output_base_dir: str, move_files: bool = True):
        """
        处理文件夹中的所有图片
        
        Args:
            input_dir: 输入图片文件夹
            output_base_dir: 输出分类基础文件夹
            move_files: 是否移动文件（True）还是复制（False）
        """
        input_path = Path(input_dir)
        output_path = Path(output_base_dir)
        
        if not input_path.exists():
            logger.error(f"输入文件夹不存在: {input_dir}")
            return
        
        # 创建输出文件夹
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_path.rglob('*') 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图片文件")
            return
        
        logger.info(f"🔍 找到 {len(image_files)} 张图片需要处理")
        
        # 处理每张图片
        for image_file in tqdm(image_files, desc="处理图片"):
            try:
                # 检测车牌
                plate_count, detections = self.detect_license_plates(str(image_file))
                
                # 确定分类
                category = self.classify_by_plate_count(plate_count)
                
                # 创建目标文件夹
                target_dir = output_path / category
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 移动或复制文件
                target_file = target_dir / image_file.name
                
                # 如果目标文件已存在，添加序号
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                if move_files:
                    shutil.move(str(image_file), str(target_file))
                    operation = "移动"
                else:
                    shutil.copy2(str(image_file), str(target_file))
                    operation = "复制"
                
                # 更新统计
                self.stats['total_processed'] += 1
                if category not in self.stats['classification_counts']:
                    self.stats['classification_counts'][category] = 0
                self.stats['classification_counts'][category] += 1
                
                # 记录详细信息
                logger.debug(f"{operation} {image_file.name} -> {category} (检测到 {plate_count} 个车牌)")
                
                # 如果检测到车牌，记录详细信息
                if detections:
                    for i, detection in enumerate(detections):
                        conf = detection['confidence']
                        bbox = detection['bbox']
                        logger.debug(f"  车牌 {i+1}: 置信度={conf:.3f}, 位置=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]")
                
            except Exception as e:
                logger.error(f"处理图片失败 {image_file}: {e}")
                continue
        
        # 输出统计信息
        self._print_stats()
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("📊 处理统计信息")
        logger.info("="*60)
        logger.info(f"总处理图片数: {self.stats['total_processed']}")
        logger.info(f"总检测车牌数: {self.stats['total_plates_detected']}")
        logger.info(f"平均每张图片车牌数: {self.stats['total_plates_detected']/max(1, self.stats['total_processed']):.2f}")
        logger.info("\n分类统计:")
        
        for category, count in sorted(self.stats['classification_counts'].items()):
            percentage = (count / max(1, self.stats['total_processed'])) * 100
            logger.info(f"  {category}: {count} 张 ({percentage:.1f}%)")
        
        logger.info("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="车牌检测分类器")
    parser.add_argument("--model", type=str, 
                       default="/home/zhiqics/sanjian/predata/models/license_plate_detector.pt",
                       help="YOLO车牌检测模型路径")
    parser.add_argument("--input", type=str,
                       default="/home/zhiqics/sanjian/predata/output_frames71/processed_output/unique_high_score_images/0_faces",
                       help="输入图片文件夹路径")
    parser.add_argument("--output", type=str,
                       default="/home/zhiqics/sanjian/predata/output_frames71/processed_output/unique_high_score_images/0_faces",
                       help="输出分类文件夹路径")
    parser.add_argument("--confidence", type=float, default=0.4,
                       help="车牌检测置信度阈值 (默认: 0.3)")
    parser.add_argument("--copy", action="store_true",
                       help="复制文件而不是移动文件")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not YOLO_AVAILABLE:
        logger.error("请先安装ultralytics: pip install ultralytics")
        return
    
    # 检查模型文件
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return
    
    # 检查输入文件夹
    if not os.path.exists(args.input):
        logger.error(f"输入文件夹不存在: {args.input}")
        return
    
    logger.info("🚀 启动车牌检测分类器")
    logger.info(f"📂 输入文件夹: {args.input}")
    logger.info(f"📁 输出文件夹: {args.output}")
    logger.info(f"🎯 模型文件: {args.model}")
    logger.info(f"🎚️ 置信度阈值: {args.confidence}")
    logger.info(f"📋 操作模式: {'复制' if args.copy else '移动'}")
    
    try:
        # 创建分类器
        classifier = LicensePlateClassifier(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # 处理图片
        classifier.process_images(
            input_dir=args.input,
            output_base_dir=args.output,
            move_files=not args.copy
        )
        
        logger.info("✅ 车牌分类完成!")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()
