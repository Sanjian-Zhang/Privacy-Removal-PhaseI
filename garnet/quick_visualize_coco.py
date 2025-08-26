#!/usr/bin/env python3
"""
快速可视化COCO标注的脚本
只生成可视化结果，不进行文本去除处理
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
    """加载COCO格式的标注文件"""
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
    """从文件名提取图片ID"""
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
    """快速可视化标注结果"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    image_name = Path(image_path).stem
    image_id = extract_image_id_from_filename(image_name)
    
    img_annotations = [ann for ann in annotations if ann.get('image_id') == image_id]
    
    colors = {
        1: (255, 0, 0),    # 蓝色 - person_or_vehicle
        2: (0, 255, 0),    # 绿色 - object
        3: (0, 0, 255)     # 红色 - text_or_license_plate
    }
    
    # 统计每种类别的数量
    category_counts = {1: 0, 2: 0, 3: 0}
    
    for ann in img_annotations:
        cat_id = ann.get('category_id', 0)
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        color = colors.get(cat_id, (128, 128, 128))
        
        # 绘制分割多边形
        segmentation = ann.get('segmentation', [])
        if segmentation and len(segmentation) > 0:
            polygon = segmentation[0]
            if len(polygon) >= 6:
                points = np.array([(int(polygon[i]), int(polygon[i+1])) 
                                 for i in range(0, len(polygon), 2)], np.int32)
                cv2.polylines(image, [points], True, color, 2)
                # 填充半透明
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], color)
                image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        
        # 绘制边界框
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # 添加类别标签
            cat_name = {1: 'Person/Vehicle', 2: 'Object', 3: 'Text/Plate'}.get(cat_id, 'Unknown')
            cv2.putText(image, cat_name, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 在图片上添加统计信息
    stats_text = f"P/V: {category_counts[1]}, OBJ: {category_counts[2]}, TXT: {category_counts[3]}"
    cv2.putText(image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # 保存图片
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
    """分析标注数据的统计信息"""
    annotations = coco_data.get('annotations', [])
    
    print(f"\n=== 标注数据分析 ===")
    print(f"总标注数量: {len(annotations)}")
    
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
    
    print(f"涉及图片数量: {len(image_ids)}")
    print(f"类别分布:")
    for cat_id, count in category_counts.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  类别 {cat_id} ({cat_name}): {count} 个标注")
    
    # 统计包含每种类别的图片数量
    images_with_category = {1: 0, 2: 0, 3: 0}
    for img_id, counts in image_annotation_counts.items():
        for cat_id in [1, 2, 3]:
            if counts[cat_id] > 0:
                images_with_category[cat_id] += 1
    
    print(f"\n包含各类别的图片数量:")
    for cat_id, count in images_with_category.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  包含类别 {cat_id} ({cat_name}): {count} 张图片")
    
    return category_counts, image_ids, image_annotation_counts

def main():
    parser = argparse.ArgumentParser(description="快速可视化COCO标注")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="训练图片目录")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO标注文件")
    parser.add_argument("--output_dir", default="/home/zhiqics/sanjian/dataset/images/QuickVisualized",
                       help="输出目录")
    parser.add_argument("--max_images", type=int, default=None, help="处理的最大图片数量")
    parser.add_argument("--format", default="JPEG", choices=["PNG", "JPEG"], 
                       help="输出图片格式")
    parser.add_argument("--only_with_text", action="store_true", 
                       help="只处理包含文本/车牌标注的图片")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COCO标注快速可视化工具")
    print(f"输出格式: {args.format}")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载标注文件
    coco_data = load_coco_annotations(args.annotations)
    if not coco_data:
        print("无法加载标注文件")
        return
    
    # 分析标注数据
    category_counts, image_ids, image_annotation_counts = analyze_annotations(coco_data)
    annotations = coco_data.get('annotations', [])
    
    # 获取图片文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(args.images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    # 如果只处理包含文本的图片
    if args.only_with_text:
        filtered_files = []
        for image_file in image_files:
            image_name = Path(image_file).stem
            image_id = extract_image_id_from_filename(image_name)
            if image_id in image_annotation_counts and image_annotation_counts[image_id][3] > 0:
                filtered_files.append(image_file)
        image_files = filtered_files
        print(f"筛选后包含文本/车牌标注的图片: {len(image_files)} 张")
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\n处理 {len(image_files)} 张图片...")
    
    processed_count = 0
    total_annotations = 0
    
    # 处理每张图片
    for image_file in tqdm(image_files, desc="可视化标注"):
        image_path = os.path.join(args.images_dir, image_file)
        image_name = Path(image_file).stem
        
        ext = '.png' if args.format == 'PNG' else '.jpg'
        output_path = os.path.join(args.output_dir, f"{image_name}_annotated{ext}")
        
        ann_count = visualize_annotations_fast(image_path, annotations, output_path, args.format)
        if ann_count > 0:
            processed_count += 1
            total_annotations += ann_count
    
    print(f"\n处理完成!")
    print(f"处理图片数量: {processed_count}")
    print(f"总标注数量: {total_annotations}")
    print(f"平均每张图片标注数量: {total_annotations/processed_count if processed_count > 0 else 0:.1f}")
    print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
