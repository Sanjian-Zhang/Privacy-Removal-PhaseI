#!/usr/bin/env python3
"""
Quick script for visualizing COCO annotations
Only generates visualization results, no text removal processing
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image

def load_coco_annotations(annotation_file):
    """Load COCO format annotation file"""
    print(f"Loading annotations from {annotation_file}...")
    
    with open(annotation_file, 'r') as f:
        content = f.read()
        
    try:
        if content.strip().startswith('{'):
            data = json.loads(content)
            return data
        else:
            annotations = []
            lines = content.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('e) zhiqics'):
                    if '"id":' in line and '"image_id":' in line:
                        try:
                            start = line.find('{"id":')
                            if start != -1:
                                brace_count = 0
                                end = start
                                for i, char in enumerate(line[start:], start):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            end = i + 1
                                            break
                                
                                json_str = line[start:end]
                                annotation = json.loads(json_str)
                                annotations.append(annotation)
                        except json.JSONDecodeError:
                            continue
            
            return {
                'annotations': annotations,
                'images': [],
                'categories': [
                    {'id': 1, 'name': 'person_or_vehicle'},
                    {'id': 2, 'name': 'object'},
                    {'id': 3, 'name': 'text_or_license_plate'}
                ]
            }
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def extract_image_id_from_filename(filename):
    """Extract image ID from filename"""
    try:
        if 'frame_' in filename:
            number_part = filename.split('frame_')[1].split('.')[0]
            return int(number_part.lstrip('0') or '0')
        else:
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return int(numbers[0])
    except:
        pass
    return 0

def visualize_annotations_fast(image_path, annotations, output_path, save_format='JPEG'):
    """Fast visualization of annotation results"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    image_name = Path(image_path).stem
    image_id = extract_image_id_from_filename(image_name)
    
    img_annotations = [ann for ann in annotations if ann.get('image_id') == image_id]
    
    colors = {
        1: (255, 0, 0),    # blue - person_or_vehicle
        2: (0, 255, 0),    # green - object
        3: (0, 0, 255)     # red - text_or_license_plate
    }
    
    category_counts = {1: 0, 2: 0, 3: 0}
    
    for ann in img_annotations:
        cat_id = ann.get('category_id', 0)
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        color = colors.get(cat_id, (128, 128, 128))
        
        segmentation = ann.get('segmentation', [])
        if segmentation and len(segmentation) > 0:
            polygon = segmentation[0]
            if len(polygon) >= 6:
                points = np.array([(int(polygon[i]), int(polygon[i+1])) 
                                 for i in range(0, len(polygon), 2)], np.int32)
                cv2.polylines(image, [points], True, color, 2)
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], color)
                image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            cat_name = {1: 'Person/Vehicle', 2: 'Object', 3: 'Text/Plate'}.get(cat_id, 'Unknown')
            cv2.putText(image, cat_name, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    stats_text = f"P/V: {category_counts[1]}, OBJ: {category_counts[2]}, TXT: {category_counts[3]}"
    cv2.putText(image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    if save_format.upper() == 'PNG':
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        if not output_path.lower().endswith('.jpg'):
            output_path = output_path.rsplit('.', 1)[0] + '.jpg'
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return len(img_annotations)

def analyze_annotations(coco_data):
    """Analyze statistical information of annotation data"""
    annotations = coco_data.get('annotations', [])
    
    print(f"\n=== Annotation Data Analysis ===")
    print(f"Total annotations: {len(annotations)}")
    
    category_counts = {}
    image_ids = set()
    image_annotation_counts = {}
    
    for ann in annotations:
        cat_id = ann.get('category_id', 0)
        image_id = ann.get('image_id')
        
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        image_ids.add(image_id)
        
        if image_id not in image_annotation_counts:
            image_annotation_counts[image_id] = {1: 0, 2: 0, 3: 0}
        image_annotation_counts[image_id][cat_id] = image_annotation_counts[image_id].get(cat_id, 0) + 1
    
    print(f"Images involved: {len(image_ids)}")
    print(f"Category distribution:")
    for cat_id, count in category_counts.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  Category {cat_id} ({cat_name}): {count} annotations")
    
    images_with_category = {1: 0, 2: 0, 3: 0}
    for img_id, counts in image_annotation_counts.items():
        for cat_id in [1, 2, 3]:
            if counts[cat_id] > 0:
                images_with_category[cat_id] += 1
    
    print(f"\nImages containing each category:")
    for cat_id, count in images_with_category.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  Images with category {cat_id} ({cat_name}): {count} images")
    
    return category_counts, image_ids, image_annotation_counts

def main():
    parser = argparse.ArgumentParser(description="Quick visualization of COCO annotations")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="Training image directory")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO annotation file")
    parser.add_argument("--output_dir", default="/home/zhiqics/sanjian/dataset/images/QuickVisualized",
                       help="Output directory")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--format", default="JPEG", choices=["PNG", "JPEG"], 
                       help="Output image format")
    parser.add_argument("--only_with_text", action="store_true", 
                       help="Only process images containing text/license plate annotations")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COCO Annotation Quick Visualization Tool")
    print(f"Output format: {args.format}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    coco_data = load_coco_annotations(args.annotations)
    if not coco_data:
        print("Unable to load annotation file")
        return
    
    category_counts, image_ids, image_annotation_counts = analyze_annotations(coco_data)
    annotations = coco_data.get('annotations', [])
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(args.images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if args.only_with_text:
        filtered_files = []
        for image_file in image_files:
            image_name = Path(image_file).stem
            image_id = extract_image_id_from_filename(image_name)
            if image_id in image_annotation_counts and image_annotation_counts[image_id][3] > 0:
                filtered_files.append(image_file)
        image_files = filtered_files
        print(f"Filtered images containing text/license plate annotations: {len(image_files)} images")
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    processed_count = 0
    total_annotations = 0
    
    for image_file in tqdm(image_files, desc="Visualizing annotations"):
        image_path = os.path.join(args.images_dir, image_file)
        image_name = Path(image_file).stem
        
        ext = '.png' if args.format == 'PNG' else '.jpg'
        output_path = os.path.join(args.output_dir, f"{image_name}_annotated{ext}")
        
        ann_count = visualize_annotations_fast(image_path, annotations, output_path, args.format)
        if ann_count > 0:
            processed_count += 1
            total_annotations += ann_count
    
    print(f"\nProcessing completed!")
    print(f"Processed images: {processed_count}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations/processed_count if processed_count > 0 else 0:.1f}")
    print(f"Results saved at: {args.output_dir}")

if __name__ == "__main__":
    main()
