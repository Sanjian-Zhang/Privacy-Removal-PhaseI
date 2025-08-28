#!/usr/bin/env python3
"""
Comprehensive script for processing training images based on COCO annotation files
Supports multiple processing methods: object detection, text removal, face anonymization, etc.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

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

def analyze_annotations(coco_data):
    """Analyze statistical information of annotation data"""
    annotations = coco_data.get('annotations', [])
    
    print(f"\n=== Annotation Data Analysis ===")
    print(f"Total annotations: {len(annotations)}")
    
    category_counts = {}
    image_ids = set()
    
    for ann in annotations:
        cat_id = ann.get('category_id', 0)
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        image_ids.add(ann.get('image_id'))
    
    print(f"Images involved: {len(image_ids)}")
    print(f"Category distribution:")
    for cat_id, count in category_counts.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  Category {cat_id} ({cat_name}): {count} annotations")
    
    return category_counts, image_ids

def create_text_regions_for_garnet(annotations, image_path, output_txt_dir):
    """Create text region files for GaRNet"""
    image_name = Path(image_path).stem
    image_id_from_name = extract_image_id_from_filename(image_name)
    
    text_annotations = [ann for ann in annotations if 
                       ann.get('image_id') == image_id_from_name and 
                       ann.get('category_id') == 3]
    
    txt_file = os.path.join(output_txt_dir, f"{image_name}.txt")
    
    with open(txt_file, 'w') as f:
        for ann in text_annotations:
            segmentation = ann.get('segmentation', [])
            if segmentation and len(segmentation) > 0:
                polygon = segmentation[0]
                if len(polygon) >= 8:
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    
                    if len(points) >= 4:
                        if len(points) == 4:
                            selected_points = points
                        else:
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            
                            min_x_idx = x_coords.index(min(x_coords))
                            max_x_idx = x_coords.index(max(x_coords))
                            min_y_idx = y_coords.index(min(y_coords))
                            max_y_idx = y_coords.index(max(y_coords))
                            
                            selected_points = [
                                points[min_x_idx],
                                points[min_y_idx],
                                points[max_x_idx],
                                points[max_y_idx]
                            ]
                        
                        line = ','.join([f"{int(p[0])},{int(p[1])}" for p in selected_points])
                        f.write(line + '\n')
            else:
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    points = [
                        (int(x), int(y)),
                        (int(x + w), int(y)),
                        (int(x + w), int(y + h)),
                        (int(x), int(y + h))
                    ]
                    line = ','.join([f"{p[0]},{p[1]}" for p in points])
                    f.write(line + '\n')
    
    return len(text_annotations)

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

def visualize_annotations(image_path, annotations, output_path):
    """Visualize annotation results"""
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
    
    for ann in img_annotations:
        cat_id = ann.get('category_id', 0)
        color = colors.get(cat_id, (128, 128, 128))
        
        segmentation = ann.get('segmentation', [])
        if segmentation and len(segmentation) > 0:
            polygon = segmentation[0]
            if len(polygon) >= 6:
                points = np.array([(int(polygon[i]), int(polygon[i+1])) 
                                 for i in range(0, len(polygon), 2)], np.int32)
                cv2.polylines(image, [points], True, color, 2)
        
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            cat_name = {1: 'P/V', 2: 'OBJ', 3: 'TXT'}.get(cat_id, 'UNK')
            cv2.putText(image, cat_name, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif output_path.lower().endswith('.png'):
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    return len(img_annotations)

def process_with_garnet(image_dir, txt_dir, output_dir):
    """Process text removal using GaRNet"""
    garnet_path = "/home/zhiqics/sanjian/baseline/garnet"
    
    if not os.path.exists(os.path.join(garnet_path, "WEIGHTS/GaRNet/saved_model.pth")):
        print("GaRNet model not found, skipping text removal processing")
        return False
    
    try:
        cmd = [
            "bash", "-c",
            f"cd {garnet_path} && "
            f"source garnet_env/bin/activate && "
            f"cd CODE && "
            f"python inference.py --gpu "
            f"--image_path {image_dir} "
            f"--box_path {txt_dir} "
            f"--result_path {output_dir}"
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ GaRNet text removal processing completed")
            return True
        else:
            print(f"✗ GaRNet processing failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running GaRNet: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process training images with COCO annotations")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="Training image directory")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO annotation file")
    parser.add_argument("--output_base", default="/home/zhiqics/sanjian/dataset/images/Processed",
                       help="Output base directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization results")
    parser.add_argument("--text_removal", action="store_true", help="Use GaRNet for text removal")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COCO Annotation Image Processing Tool")
    print("="*60)
    
    coco_data = load_coco_annotations(args.annotations)
    if not coco_data:
        print("Unable to load annotation file")
        return
    
    category_counts, image_ids = analyze_annotations(coco_data)
    annotations = coco_data.get('annotations', [])
    
    output_dirs = {
        'visualized': os.path.join(args.output_base, 'visualized'),
        'garnet_input_txt': os.path.join(args.output_base, 'garnet_input_txt'),
        'garnet_input_img': os.path.join(args.output_base, 'garnet_input_img'),
        'garnet_output': os.path.join(args.output_base, 'garnet_output')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(args.images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    processed_count = 0
    text_regions_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.images_dir, image_file)
        image_name = Path(image_file).stem
        
        if args.visualize:
            vis_output = os.path.join(output_dirs['visualized'], f"{image_name}_annotated.jpg")
            ann_count = visualize_annotations(image_path, annotations, vis_output)
            if ann_count > 0:
                processed_count += 1
        
        if args.text_removal:
            garnet_img_path = os.path.join(output_dirs['garnet_input_img'], image_file)
            shutil.copy2(image_path, garnet_img_path)
            
            text_count = create_text_regions_for_garnet(
                annotations, image_path, output_dirs['garnet_input_txt']
            )
            text_regions_count += text_count
    
    print(f"\nProcessing completed!")
    print(f"Processed images: {processed_count}")
    
    if args.text_removal and text_regions_count > 0:
        print(f"Text regions detected: {text_regions_count}")
        print("Starting GaRNet text removal processing...")
        
        success = process_with_garnet(
            output_dirs['garnet_input_img'],
            output_dirs['garnet_input_txt'],
            output_dirs['garnet_output']
        )
        
        if success:
            processed_files = os.listdir(output_dirs['garnet_output'])
            print(f"✓ Text removal completed, generated {len(processed_files)} files")
    
    print(f"\nResults saved at:")
    for name, path in output_dirs.items():
        file_count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {name}: {path} ({file_count} files)")

if __name__ == "__main__":
    main()
