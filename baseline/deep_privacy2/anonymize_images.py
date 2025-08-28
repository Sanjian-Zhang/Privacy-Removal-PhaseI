#!/usr/bin/env python3
"""
Simplified image anonymization script to avoid moviepy dependency issues
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


def anonymize_image(image_path, output_path, anonymizer, visualize=False):
    """Anonymize single image - maintain original size and optimized parameters"""
    image = load_image(image_path)
    if image is None:
        return False
    
    original_size = (image.shape[1], image.shape[0])
    
    try:
        import hashlib
        md5_ = hashlib.md5(image).hexdigest()
        image_tensor = utils.im2torch(image, to_float=False, normalize=False)[0]
        
        synthesis_kwargs = {
            'cache_id': md5_,
            'multi_modal_truncation': False,
            'amp': False,
            'truncation_value': 0.6
        }
        
        anonymized = anonymizer(image_tensor, **synthesis_kwargs)
        
        anonymized = utils.im2numpy(anonymized)
        
        if visualize:
            cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Anonymized', cv2.cvtColor(anonymized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
        if output_path:
            anonymized_pil = Image.fromarray(anonymized)
            
            if anonymized_pil.size != original_size:
                try:
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
                except AttributeError:
                    anonymized_pil = anonymized_pil.resize(original_size, resample=Image.LANCZOS)
            
            output_path = Path(output_path)
            filename_parts = output_path.stem.split('.')
            base_name = filename_parts[0]
            extension = output_path.suffix
            
            ADD_OPTIMIZATION_TAG = True
            
            if ADD_OPTIMIZATION_TAG:
                new_filename = f"{base_name}_opt_tv06{extension}"
                new_output_path = output_path.parent / new_filename
            else:
                new_output_path = output_path
            
            anonymized_pil.save(new_output_path, optimize=False, quality=95)
            logger.log(f"Saved anonymized image to: {new_output_path} (size: {anonymized_pil.size})")
        
        return True
    except Exception as e:
        import traceback
        logger.log(f"Error processing image {image_path}: {e}")
        logger.log(f"Traceback: {traceback.format_exc()}")
        return False


def anonymize_directory(input_dir, output_dir, anonymizer, visualize=False):
    """Batch anonymize images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
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
        
        if anonymize_image(image_path, output_file, anonymizer, visualize):
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
    "--seed", 
    default=0, 
    type=int, 
    help="Random seed"
)
def main(config_path, input_path, output_path, visualize, detection_score_threshold, seed):
    """Simplified image anonymization tool"""
    
    tops.set_seed(seed)
    
    cfg = utils.load_config(config_path)
    cfg.detector.score_threshold = detection_score_threshold
    utils.print_config(cfg)
    
    logger.log("Initializing anonymizer...")
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if input_path.is_dir():
        anonymize_directory(input_path, output_path, anonymizer, visualize)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anonymize_image(input_path, output_path, anonymizer, visualize)


if __name__ == "__main__":
    main()
