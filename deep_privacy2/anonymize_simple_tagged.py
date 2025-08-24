#!/usr/bin/env python3
"""
Simple version: Add filename identification based on current processing
This script can be used for new batch processing, adding parameter tags to output files
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
    """Load image file"""
    try:
        im = Image.open(image_path).convert("RGB")
        im = _apply_exif_orientation(im)
        return np.array(im)
    except Exception as e:
        logger.log(f"Error loading image {image_path}: {e}")
        return None


def anonymize_image(image_path, output_path, anonymizer, add_tag=True, visualize=False):
    """Anonymize single image - maintain original size and optionally add identification"""
    image = load_image(image_path)
    if image is None:
        return False
    
    # Record original dimensions
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    try:
        # Use dp2's utility function to correctly convert image
        import hashlib
        md5_ = hashlib.md5(image).hexdigest()
        image_tensor = utils.im2torch(image, to_float=False, normalize=False)[0]
        
        # Set necessary synthesis parameters - optimized version
        synthesis_kwargs = {
            'cache_id': md5_,
            'multi_modal_truncation': False,  # Set to True to increase diversity
            'amp': False,
            'truncation_value': 0.6  # Optimization parameter: more realistic effects
        }
        
        # Perform anonymization
        anonymized = anonymizer(image_tensor, **synthesis_kwargs)
        
        # Convert back to numpy array
        anonymized = utils.im2numpy(anonymized)
        
        if visualize:
            # Show results
            cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Anonymized', cv2.cvtColor(anonymized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
        # Save results - optional tag addition
        if output_path:
            anonymized_pil = Image.fromarray(anonymized)
            
            # If dimensions differ, resize back to original size
            if anonymized_pil.size != original_size:
                # Use high-quality resampling
                try:
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
                except AttributeError:
                    # Compatibility with older PIL versions
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.LANCZOS)
            
            # Determine output path
            if add_tag:
                # Add optimization tag to filename
                output_path = Path(output_path)
                filename_parts = output_path.stem.split('.')
                base_name = filename_parts[0]
                extension = output_path.suffix
                
                # Create new filename with optimization tag
                # Format: {original_name}_optimized_tv06_dt30.jpg
                new_filename = f"{base_name}_optimized_tv06_dt30{extension}"
                final_output_path = output_path.parent / new_filename
            else:
                final_output_path = output_path
            
            # High-quality save
            anonymized_pil.save(final_output_path, optimize=False, quality=95)
            logger.log(f"Saved anonymized image to: {final_output_path} (size: {anonymized_pil.size})")
        
        return True
    except Exception as e:
        import traceback
        logger.log(f"Error processing image {image_path}: {e}")
        logger.log(f"Traceback: {traceback.format_exc()}")
        return False


def anonymize_directory(input_dir, output_dir, anonymizer, add_tag=True, visualize=False):
    """Batch anonymize images in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
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
        
        if anonymize_image(image_path, output_file, anonymizer, add_tag, visualize):
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
    help="Detection threshold"
)
@click.option(
    "--add-tag/--no-tag",
    default=True,
    help="Add optimization tags to filenames"
)
@click.option(
    "--seed", 
    default=0, 
    type=int, 
    help="Random seed"
)
def main(config_path, input_path, output_path, visualize, detection_score_threshold, add_tag, seed):
    """
    Simplified image anonymization tool - Optional filename tagging
    
    If tagging is enabled (--add-tag), output filename format:
    {original_name}_optimized_tv06_dt30.jpg
    
    Parameter explanation:
    - tv06: truncation_value=0.6 (optimized truncation value)
    - dt30: detection_threshold=0.3 (detection threshold)
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
    
    # 显示配置信息
    logger.log(f"Using optimized parameters:")
    logger.log(f"  - Detection threshold: {detection_score_threshold}")
    logger.log(f"  - Truncation value: 0.6 (optimized)")
    logger.log(f"  - Add filename tags: {add_tag}")
    logger.log(f"  - Image quality: 95")
    
    # 处理图像
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if input_path.is_dir():
        anonymize_directory(input_path, output_path, anonymizer, add_tag, visualize)
    else:
        # 单个文件处理
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anonymize_image(input_path, output_path, anonymizer, add_tag, visualize)


if __name__ == "__main__":
    main()
