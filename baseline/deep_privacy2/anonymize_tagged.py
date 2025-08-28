#!/usr/bin/env python3
"""
Tagged optimized image anonymization script
Processed image filenames will contain parameter tags for distinguishing effects of different settings
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


def create_filename_with_tags(original_path, output_dir, 
                             detection_threshold, truncation_value, 
                             custom_tag="dp2_opt"):
    """
    创建带有参数标识的文件名
    格式: {原始名称}_{custom_tag}_dt{detection_threshold}_tv{truncation_value}.{扩展名}
    """
    original_path = Path(original_path)
    output_dir = Path(output_dir)
    
    base_name = original_path.stem
    extension = original_path.suffix
    
    # 格式化参数标识
    dt_str = f"dt{detection_threshold:.2f}".replace("0.", "")
    tv_str = f"tv{truncation_value:.2f}".replace("0.", "")
    
    # Create new filename
    new_filename = f"{base_name}_{custom_tag}_{dt_str}_{tv_str}{extension}"
    return output_dir / new_filename


def anonymize_image(image_path, output_path, anonymizer, 
                   detection_threshold=0.3, truncation_value=0.6,
                   multi_modal_truncation=False, 
                   preserve_original_size=True,
                   quality=95, custom_tag="dp2_opt",
                   visualize=False):
    """
    Anonymize a single image and add parameter tags
    """
    image = load_image(image_path)
    if image is None:
        return False
    
    # Record original dimensions
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    try:
        # Use dp2's utility functions to properly convert image
        import hashlib
        md5_ = hashlib.md5(image).hexdigest()
        image_tensor = utils.im2torch(image, to_float=False, normalize=False)[0]
        
        # Set synthesis parameters
        synthesis_kwargs = {
            'cache_id': md5_,
            'multi_modal_truncation': multi_modal_truncation,
            'amp': False,
            'truncation_value': truncation_value
        }
        
        # Perform anonymization processing
        anonymized = anonymizer(image_tensor, **synthesis_kwargs)
        
        # Convert back to numpy array
        anonymized = utils.im2numpy(anonymized)
        
        if visualize:
            # Display results
            cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Anonymized', cv2.cvtColor(anonymized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
        # Save results - with parameter tags
        if output_path:
            anonymized_pil = Image.fromarray(anonymized)
            
            # If need to preserve original size
            if preserve_original_size and anonymized_pil.size != original_size:
                try:
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
                except AttributeError:
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.LANCZOS)
            
            # Create filename with tags
            tagged_output_path = create_filename_with_tags(
                image_path, output_path.parent if output_path.is_file() else output_path,
                detection_threshold, truncation_value, custom_tag
            )
            
            # Ensure output directory exists
            tagged_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # High quality save
            anonymized_pil.save(tagged_output_path, optimize=False, quality=quality)
            logger.log(f"Saved: {tagged_output_path.name} (size: {anonymized_pil.size})")
        
        return True
    except Exception as e:
        import traceback
        logger.log(f"Error processing {image_path}: {e}")
        logger.log(f"Traceback: {traceback.format_exc()}")
        return False


def anonymize_directory(input_dir, output_dir, anonymizer, **kwargs):
    """Batch anonymize images in directory"""
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
        
        if anonymize_image(image_path, output_path, anonymizer, **kwargs):
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
    "--truncation-value", "--tv",
    default=0.6,
    type=click.FloatRange(0, 1),
    help="Truncation value (lower=more conservative)"
)
@click.option(
    "--multi-modal-truncation", "--mmt",
    default=False,
    is_flag=True,
    help="Enable multi-modal truncation"
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
    "--custom-tag", "--tag",
    default="dp2_opt",
    help="Custom tag for output filenames"
)
@click.option(
    "--seed", 
    default=0, 
    type=int, 
    help="Random seed"
)
def main(config_path, input_path, output_path, visualize, detection_score_threshold, 
         truncation_value, multi_modal_truncation, preserve_size, quality, 
         custom_tag, seed):
    """
    Optimized image anonymization tool - with parameter tags
    
    Output filename format: {original_name}_{tag}_dt{detection_threshold}_tv{truncation_value}.jpg
    
    Example output filenames:
    - frame_00001_dp2_opt_dt30_tv60.jpg
    - frame_00002_high_quality_dt25_tv55.jpg
    """
    
    # Set random seed
    tops.set_seed(seed)
    
    # Load configuration
    cfg = utils.load_config(config_path)
    cfg.detector.score_threshold = detection_score_threshold
    utils.print_config(cfg)
    
    # Initialize anonymizer
    logger.log("Initializing anonymizer...")
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    
    # Display configuration information
    logger.log(f"Configuration:")
    logger.log(f"  - Detection threshold: {detection_score_threshold}")
    logger.log(f"  - Truncation value: {truncation_value}")
    logger.log(f"  - Multi-modal truncation: {multi_modal_truncation}")
    logger.log(f"  - Custom tag: {custom_tag}")
    logger.log(f"  - Output quality: {quality}")
    
    # Process images
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Pass parameters
    kwargs = {
        'detection_threshold': detection_score_threshold,
        'truncation_value': truncation_value,
        'multi_modal_truncation': multi_modal_truncation,
        'preserve_original_size': preserve_size,
        'quality': quality,
        'custom_tag': custom_tag,
        'visualize': visualize
    }
    
    if input_path.is_dir():
        anonymize_directory(input_path, output_path, anonymizer, **kwargs)
    else:
        # Single file processing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anonymize_image(input_path, output_path, anonymizer, **kwargs)


if __name__ == "__main__":
    main()
