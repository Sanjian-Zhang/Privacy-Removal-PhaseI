#!/usr/bin/env python3
"""
Demo script showing how to use Garnet model with COCO annotation labels
This is a simplified example script demonstrating the main processing workflow
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def load_coco_data(annotation_file):
    """Load COCO annotation data"""
    with open(annotation_file, 'r') as f:
        data = json.loads(f.read())
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    images = {img['id']: img for img in data['images']}
    
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    return categories, images, annotations_by_image

def filter_text_annotations(annotations, text_category_id=3):
    """Filter text category annotations"""
    return [ann for ann in annotations if ann['category_id'] == text_category_id]

def create_garnet_input(image, text_annotations, output_txt_path, output_img_path):
    """Create input files for Garnet model"""
    height, width = image.shape[:2]
    
    with open(output_txt_path, 'w') as f:
        for i, ann in enumerate(text_annotations):
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            f.write(f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2}\n")
    
    cv2.imwrite(output_img_path, image)
    print(f"Created Garnet input: {output_txt_path}, {output_img_path}")

def visualize_annotations(image, annotations, categories, output_path):
    """Visualize annotations"""
    vis_image = image.copy()
    
    colors = {
        1: (0, 255, 0),    # face/vehicle - green
        2: (255, 0, 0),    # object - blue  
        3: (0, 0, 255),    # text/license plate - red
    }
    
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        category_id = ann['category_id']
        category_name = categories.get(category_id, f"unknown{category_id}")
        
        color = colors.get(category_id, (128, 128, 128))
        
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        label = f"{category_name} (ID:{category_id})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to: {output_path}")

def process_single_image(image_path, image_id, annotations, categories, output_dir):
    """Process single image"""
    print(f"\n=== Processing image: {os.path.basename(image_path)} ===")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    
    print(f"Image size: {image.shape}")
    print(f"Annotation count: {len(annotations)}")
    
    category_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f"unknown{cat_id}")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    print("Category distribution:", category_counts)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    visualize_annotations(image, annotations, categories, vis_path)
    
    text_annotations = filter_text_annotations(annotations, text_category_id=3)
    if text_annotations:
        print(f"Found {len(text_annotations)} text regions")
        
        txt_path = os.path.join(output_dir, f"{base_name}_text_regions.txt")
        img_path = os.path.join(output_dir, f"{base_name}_garnet_input.jpg")
        create_garnet_input(image, text_annotations, txt_path, img_path)
        
        print("Garnet processing command example:")
        print(f"python garnet_inference.py --input_image {img_path} --input_txt {txt_path} --output_dir {output_dir}")
    else:
        print("No text regions found")

def main():
    parser = argparse.ArgumentParser(description="COCO annotation + Garnet model demo")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="Image directory path")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO annotation file path")
    parser.add_argument("--output_dir", default="/home/zhiqics/sanjian/demo_output",
                       help="Output directory")
    parser.add_argument("--image_name", default="frame_00030.jpg",
                       help="Image name to process")
    
    args = parser.parse_args()
    
    print("=== COCO Annotation + Garnet Model Processing Demo ===")
    print(f"Image directory: {args.images_dir}")
    print(f"Annotation file: {args.annotations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing image: {args.image_name}")
    
    print("\nLoading COCO annotation data...")
    categories, images, annotations_by_image = load_coco_data(args.annotations)
    
    print(f"Category definitions: {categories}")
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {sum(len(anns) for anns in annotations_by_image.values())}")
    
    target_image_id = None
    for img_id, img_info in images.items():
        if img_info['file_name'] == args.image_name:
            target_image_id = img_id
            break
    
    if target_image_id is None:
        print(f"Image not found: {args.image_name}")
        print("Available image examples:")
        for i, (img_id, img_info) in enumerate(list(images.items())[:5]):
            print(f"  {img_info['file_name']}")
        return
    
    image_annotations = annotations_by_image.get(target_image_id, [])
    image_path = os.path.join(args.images_dir, args.image_name)
    
    process_single_image(image_path, target_image_id, image_annotations, categories, args.output_dir)
    
    print(f"\nProcessing completed! Results saved at: {args.output_dir}")

if __name__ == "__main__":
    main()
