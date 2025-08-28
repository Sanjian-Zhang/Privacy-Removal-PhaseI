#!/usr/bin/env python3
"""
Process images using COCO annotations for face and license plate blurring
"""

import os
import glob
import json
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path

# Set environment variable to allow unsafe loading (for trusted models)
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

def load_coco_annotations(coco_file):
    """Load COCO format annotations"""
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from category_id to category name
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create mapping from image_id to image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    return categories, images, image_annotations

def main():
    parser = argparse.ArgumentParser(description='Process images with COCO annotations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if use_coco_labels is enabled
    if not config.get('use_coco_labels', False):
        print("COCO labels are not enabled in config. Set 'use_coco_labels: true' to use this script.")
        return
    
    coco_file = config['coco_annotation_file']
    if not os.path.exists(coco_file):
        print(f"COCO annotation file not found: {coco_file}")
        return
    
    print(f"Loading COCO annotations from: {coco_file}")
    categories, images, image_annotations = load_coco_annotations(coco_file)
    
    print(f"Found {len(categories)} categories: {list(categories.values())}")
    print(f"Found {len(images)} images")
    print(f"Found annotations for {len(image_annotations)} images")
    
    # Get class mapping from config
    class_mapping = config.get('class_mapping', {})
    print(f"Class mapping: {class_mapping}")
    
    # Create output directory
    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images from input directory
    image_files = glob.glob(os.path.join(config['images_path'], f"*{config['img_format']}"))
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    blurred_count = 0
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # Find corresponding image in COCO data
        matching_image = None
        matching_img_id = None
        for img_id, img_info in images.items():
            if img_info['file_name'] == img_name:
                matching_image = img_info
                matching_img_id = img_id
                break
        
        if matching_image is None:
            print(f"No COCO annotation found for {img_name}, copying original...")
            # Copy original image
            img = cv2.imread(img_path)
            if img is not None:
                output_path = os.path.join(output_folder, f"{img_name_no_ext}_processed.jpg")
                cv2.imwrite(output_path, img)
            processed_count += 1
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image: {img_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Get annotations for this image
        annotations = image_annotations.get(matching_img_id, [])
        
        blur_applied = False
        for ann in annotations:
            cat_name = categories[ann['category_id']]
            
            # Check if this category should be blurred
            if cat_name in class_mapping and class_mapping[cat_name] >= 0:
                # Get bounding box
                x, y, w, h = ann['bbox']
                
                # Convert to integer coordinates
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_width, int(x + w))
                y2 = min(img_height, int(y + h))
                
                # Apply blur to the region
                if x2 > x1 and y2 > y1:
                    roi = image[y1:y2, x1:x2]
                    blur_radius = config.get('blur_radius', 31)
                    # Ensure blur radius is odd
                    if blur_radius % 2 == 0:
                        blur_radius += 1
                    blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
                    image[y1:y2, x1:x2] = blurred_roi
                    blur_applied = True
                    print(f"Blurred {cat_name} in {img_name} at ({x1},{y1},{x2},{y2})")
        
        # Save processed image
        output_path = os.path.join(output_folder, f"{img_name_no_ext}_processed.jpg")
        cv2.imwrite(output_path, image)
        
        processed_count += 1
        if blur_applied:
            blurred_count += 1
        
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(image_files)} images...")
    
    print(f"Finished processing {processed_count} images")
    print(f"Applied blur to {blurred_count} images")
    print(f"Output saved in: {output_folder}")

if __name__ == "__main__":
    main()
