#!/usr/bin/env python3
"""
优化版的图像匿名化脚本，提供更多参数调整选项以获得最佳匿名化效果
"""
import cv2
import torch
import tops
import tqdm
import click
import hashlib
import numpy as np
from dp2 import utils
from PIL import Image
from tops import logger
from pathlib import Path
from typing import Optional
from tops.config import instantiate
from detectron2.data.detection_utils import _apply_exif_orientation
import os
import glob


def load_image(image_path):
    """加载图像文件"""
    try:
        im = Image.open(image_path).convert("RGB")
        im = _apply_exif_orientation(im)
        return np.array(im)
    except Exception as e:
        logger.log(f"Error loading image {image_path}: {e}")
        return None


def resize_to_original(anonymized_image, original_size):
    """将匿名化后的图像调整回原始尺寸"""
    if isinstance(anonymized_image, np.ndarray):
        anonymized_pil = Image.fromarray(anonymized_image)
    else:
        anonymized_pil = anonymized_image
    
    # 使用高质量的重采样方法
    resized = anonymized_pil.resize(original_size, resample=Image.LANCZOS)
    return resized


def anonymize_image(image_path, output_path, anonymizer, 
                   truncation_value=0.7, 
                   multi_modal_truncation=False, 
                   preserve_original_size=True,
                   quality=95,
                   visualize=False):
    """
    匿名化单个图像
    
    参数:
    - truncation_value: 截断值，控制生成的多样性 (0.0-1.0，越低越保守)
    - multi_modal_truncation: 是否使用多模态截断
    - preserve_original_size: 是否保持原始图像尺寸
    - quality: 保存质量 (1-100)
    """
    image = load_image(image_path)
    if image is None:
        return False
    
    # 记录原始尺寸
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    try:
        # 使用dp2的工具函数正确转换图像
        import hashlib
        md5_ = hashlib.md5(image).hexdigest()
        image_tensor = utils.im2torch(image, to_float=False, normalize=False)[0]
        
        # 设置synthesis参数 - 这些参数影响匿名化质量
        synthesis_kwargs = {
            'cache_id': md5_,
            'multi_modal_truncation': multi_modal_truncation,
            'amp': False,  # 自动混合精度，CPU模式下设为False
            'truncation_value': truncation_value  # 关键参数：控制生成的保守程度
        }
        
        # 进行匿名化处理
        anonymized = anonymizer(image_tensor, **synthesis_kwargs)
        
        # 转换回numpy数组
        anonymized = utils.im2numpy(anonymized)
        
        if visualize:
            # 显示结果
            cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Anonymized', cv2.cvtColor(anonymized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
        # 保存结果
        if output_path:
            anonymized_pil = Image.fromarray(anonymized)
            
            # 如果需要保持原始尺寸
            if preserve_original_size and anonymized_pil.size != original_size:
                anonymized_pil = resize_to_original(anonymized_pil, original_size)
            
            # 高质量保存
            anonymized_pil.save(output_path, optimize=False, quality=quality)
            logger.log(f"Saved anonymized image to: {output_path} (size: {anonymized_pil.size})")
        
        return True
    except Exception as e:
        import traceback
        logger.log(f"Error processing image {image_path}: {e}")
        logger.log(f"Traceback: {traceback.format_exc()}")
        return False


def anonymize_directory(input_dir, output_dir, anonymizer, **kwargs):
    """批量匿名化目录中的图像"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_path / ext)))
        image_files.extend(glob.glob(str(input_path / ext.upper())))
    
    logger.log(f"Found {len(image_files)} images to process")
    
    success_count = 0
    for image_file in tqdm.tqdm(image_files, desc="Processing images"):
        image_path = Path(image_file)
        output_file = output_path / image_path.name
        
        if anonymize_image(image_path, output_file, anonymizer, **kwargs):
            success_count += 1
    
    logger.log(f"Successfully processed {success_count}/{len(image_files)} images")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "-i", "--input_path", 
    required=True,
    help="Input directory containing images"
)
@click.option(
    "-o", "--output_path", 
    required=True,
    type=click.Path(),
    help="Output directory to save anonymized images"
)
@click.option(
    "-v", "--visualize", 
    default=False, 
    is_flag=True, 
    help="Visualize the results"
)
@click.option(
    "--detection-score-threshold", "--dst",
    default=0.3,
    type=click.FloatRange(0, 1),
    help="Detection threshold (0.1-0.9, lower=more faces detected)"
)
@click.option(
    "--truncation-value", "--tv",
    default=0.7,
    type=click.FloatRange(0, 1),
    help="Truncation value (0.0-1.0, lower=more conservative anonymization)"
)
@click.option(
    "--multi-modal-truncation", "--mmt",
    default=False,
    is_flag=True,
    help="Enable multi-modal truncation for better diversity"
)
@click.option(
    "--preserve-size", "--ps",
    default=True,
    is_flag=True,
    help="Preserve original image size"
)
@click.option(
    "--quality", "--q",
    default=95,
    type=click.IntRange(1, 100),
    help="Output image quality (1-100)"
)
@click.option(
    "--seed", 
    default=0, 
    type=int, 
    help="Random seed for reproducibility"
)
def main(config_path, input_path, output_path, visualize, detection_score_threshold, 
         truncation_value, multi_modal_truncation, preserve_size, quality, seed):
    """
    优化版图像匿名化工具
    
    参数调优建议:
    
    1. detection_score_threshold (人脸检测阈值):
       - 0.1-0.3: 检测更多人脸，包括模糊/小的人脸
       - 0.3-0.5: 平衡检测准确性和召回率 (推荐)
       - 0.5-0.9: 只检测清晰/大的人脸
    
    2. truncation_value (截断值):
       - 0.3-0.5: 非常保守，生成的人脸更真实但多样性较低
       - 0.6-0.8: 平衡真实性和多样性 (推荐)
       - 0.8-1.0: 更多样化但可能不够真实
    
    3. multi_modal_truncation:
       - False: 标准截断 (推荐用于大多数情况)
       - True: 多模态截断，增加多样性
    
    4. quality:
       - 85-95: 高质量输出 (推荐)
       - 95-100: 最高质量但文件更大
    """
    
    # 设置随机种子
    tops.set_seed(seed)
    
    # 加载配置
    cfg = utils.load_config(config_path)
    cfg.detector.score_threshold = detection_score_threshold
    utils.print_config(cfg)
    
    # 初始化匿名化器
    logger.log("Initializing anonymizer...")
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    
    # 处理图像
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 传递参数
    kwargs = {
        'truncation_value': truncation_value,
        'multi_modal_truncation': multi_modal_truncation,
        'preserve_original_size': preserve_size,
        'quality': quality,
        'visualize': visualize
    }
    
    if input_path.is_dir():
        anonymize_directory(input_path, output_path, anonymizer, **kwargs)
    else:
        # 单个文件处理
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anonymize_image(input_path, output_path, anonymizer, **kwargs)


if __name__ == "__main__":
    main()
