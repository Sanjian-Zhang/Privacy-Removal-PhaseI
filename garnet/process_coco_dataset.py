#!/usr/bin/env python3
"""
基于COCO标注文件处理训练图片的综合脚本
支持多种处理方式：目标检测、文本去除、面部匿名化等
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
    """加载COCO格式的标注文件"""
    print(f"Loading annotations from {annotation_file}...")
    
    with open(annotation_file, 'r') as f:
        # 由于文件很大，我们逐步加载
        content = f.read()
        
    # 尝试解析JSON
    try:
        # 先检查是否是完整的JSON格式
        if content.strip().startswith('{'):
            data = json.loads(content)
            return data
        else:
            # 如果不是完整JSON，尝试解析单个标注条目
            annotations = []
            lines = content.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('e) zhiqics'):
                    # 尝试提取JSON对象
                    if '"id":' in line and '"image_id":' in line:
                        try:
                            # 查找完整的JSON对象
                            start = line.find('{"id":')
                            if start != -1:
                                # 找到对应的结束括号
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
            
            # 构建简化的COCO格式数据结构
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
    """分析标注数据的统计信息"""
    annotations = coco_data.get('annotations', [])
    
    print(f"\n=== 标注数据分析 ===")
    print(f"总标注数量: {len(annotations)}")
    
    # 统计类别分布
    category_counts = {}
    image_ids = set()
    
    for ann in annotations:
        cat_id = ann.get('category_id', 0)
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        image_ids.add(ann.get('image_id'))
    
    print(f"涉及图片数量: {len(image_ids)}")
    print(f"类别分布:")
    for cat_id, count in category_counts.items():
        cat_name = {1: 'person_or_vehicle', 2: 'object', 3: 'text_or_license_plate'}.get(cat_id, 'unknown')
        print(f"  类别 {cat_id} ({cat_name}): {count} 个标注")
    
    return category_counts, image_ids

def create_text_regions_for_garnet(annotations, image_path, output_txt_dir):
    """为GaRNet创建文本区域文件"""
    image_name = Path(image_path).stem
    image_id_from_name = extract_image_id_from_filename(image_name)
    
    # 找到该图片的所有文本相关标注 (category_id == 3)
    text_annotations = [ann for ann in annotations if 
                       ann.get('image_id') == image_id_from_name and 
                       ann.get('category_id') == 3]
    
    txt_file = os.path.join(output_txt_dir, f"{image_name}.txt")
    
    with open(txt_file, 'w') as f:
        for ann in text_annotations:
            segmentation = ann.get('segmentation', [])
            if segmentation and len(segmentation) > 0:
                # 取第一个分割多边形
                polygon = segmentation[0]
                if len(polygon) >= 8:  # 至少需要4个点（8个坐标）
                    # 将多边形转换为4个角点格式
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    
                    if len(points) >= 4:
                        # 取前4个点或者选择边界点
                        if len(points) == 4:
                            selected_points = points
                        else:
                            # 如果有更多点，选择4个角点
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            
                            min_x_idx = x_coords.index(min(x_coords))
                            max_x_idx = x_coords.index(max(x_coords))
                            min_y_idx = y_coords.index(min(y_coords))
                            max_y_idx = y_coords.index(max(y_coords))
                            
                            selected_points = [
                                points[min_x_idx],  # 最左点
                                points[min_y_idx],  # 最上点
                                points[max_x_idx],  # 最右点
                                points[max_y_idx]   # 最下点
                            ]
                        
                        # 格式化为 x1,y1,x2,y2,x3,y3,x4,y4
                        line = ','.join([f"{int(p[0])},{int(p[1])}" for p in selected_points])
                        f.write(line + '\n')
            else:
                # 如果没有分割信息，使用边界框
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    # 转换为4个角点
                    points = [
                        (int(x), int(y)),                    # 左上
                        (int(x + w), int(y)),                # 右上
                        (int(x + w), int(y + h)),            # 右下
                        (int(x), int(y + h))                 # 左下
                    ]
                    line = ','.join([f"{p[0]},{p[1]}" for p in points])
                    f.write(line + '\n')
    
    return len(text_annotations)

def extract_image_id_from_filename(filename):
    """从文件名提取图片ID"""
    # 例如: frame_00030.jpg -> 30
    try:
        if 'frame_' in filename:
            number_part = filename.split('frame_')[1].split('.')[0]
            return int(number_part.lstrip('0') or '0')
        else:
            # 尝试提取任何数字
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return int(numbers[0])
    except:
        pass
    return 0

def visualize_annotations(image_path, annotations, output_path):
    """可视化标注结果"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    image_name = Path(image_path).stem
    image_id = extract_image_id_from_filename(image_name)
    
    # 找到该图片的所有标注
    img_annotations = [ann for ann in annotations if ann.get('image_id') == image_id]
    
    colors = {
        1: (255, 0, 0),    # 蓝色 - person_or_vehicle
        2: (0, 255, 0),    # 绿色 - object
        3: (0, 0, 255)     # 红色 - text_or_license_plate
    }
    
    for ann in img_annotations:
        cat_id = ann.get('category_id', 0)
        color = colors.get(cat_id, (128, 128, 128))
        
        # 绘制分割多边形
        segmentation = ann.get('segmentation', [])
        if segmentation and len(segmentation) > 0:
            polygon = segmentation[0]
            if len(polygon) >= 6:
                points = np.array([(int(polygon[i]), int(polygon[i+1])) 
                                 for i in range(0, len(polygon), 2)], np.int32)
                cv2.polylines(image, [points], True, color, 2)
        
        # 绘制边界框
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # 添加类别标签
            cat_name = {1: 'P/V', 2: 'OBJ', 3: 'TXT'}.get(cat_id, 'UNK')
            cv2.putText(image, cat_name, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save image without compression
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        # For JPEG, use maximum quality (100) to minimize compression
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif output_path.lower().endswith('.png'):
        # For PNG, use no compression
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        # Default case with maximum quality
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    return len(img_annotations)

def process_with_garnet(image_dir, txt_dir, output_dir):
    """使用GaRNet处理文本去除"""
    garnet_path = "/home/zhiqics/sanjian/baseline/garnet"
    
    if not os.path.exists(os.path.join(garnet_path, "WEIGHTS/GaRNet/saved_model.pth")):
        print("GaRNet模型未找到，跳过文本去除处理")
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
            print("✓ GaRNet文本去除处理完成")
            return True
        else:
            print(f"✗ GaRNet处理失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running GaRNet: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="处理带COCO标注的训练图片")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="训练图片目录")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO标注文件")
    parser.add_argument("--output_base", default="/home/zhiqics/sanjian/dataset/images/Processed",
                       help="输出基础目录")
    parser.add_argument("--visualize", action="store_true", help="生成可视化结果")
    parser.add_argument("--text_removal", action="store_true", help="使用GaRNet进行文本去除")
    parser.add_argument("--max_images", type=int, default=None, help="处理的最大图片数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COCO标注图片处理工具")
    print("="*60)
    
    # 加载标注文件
    coco_data = load_coco_annotations(args.annotations)
    if not coco_data:
        print("无法加载标注文件")
        return
    
    # 分析标注数据
    category_counts, image_ids = analyze_annotations(coco_data)
    annotations = coco_data.get('annotations', [])
    
    # 创建输出目录
    output_dirs = {
        'visualized': os.path.join(args.output_base, 'visualized'),
        'garnet_input_txt': os.path.join(args.output_base, 'garnet_input_txt'),
        'garnet_input_img': os.path.join(args.output_base, 'garnet_input_img'),
        'garnet_output': os.path.join(args.output_base, 'garnet_output')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取图片文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(args.images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\n处理 {len(image_files)} 张图片...")
    
    processed_count = 0
    text_regions_count = 0
    
    # 处理每张图片
    for image_file in tqdm(image_files, desc="处理图片"):
        image_path = os.path.join(args.images_dir, image_file)
        image_name = Path(image_file).stem
        
        # 可视化标注
        if args.visualize:
            vis_output = os.path.join(output_dirs['visualized'], f"{image_name}_annotated.jpg")
            ann_count = visualize_annotations(image_path, annotations, vis_output)
            if ann_count > 0:
                processed_count += 1
        
        # 为文本去除准备数据
        if args.text_removal:
            # 复制原图到GaRNet输入目录
            garnet_img_path = os.path.join(output_dirs['garnet_input_img'], image_file)
            shutil.copy2(image_path, garnet_img_path)
            
            # 创建文本区域文件
            text_count = create_text_regions_for_garnet(
                annotations, image_path, output_dirs['garnet_input_txt']
            )
            text_regions_count += text_count
    
    print(f"\n处理完成!")
    print(f"处理图片数量: {processed_count}")
    
    if args.text_removal and text_regions_count > 0:
        print(f"检测到文本区域: {text_regions_count}")
        print("开始GaRNet文本去除处理...")
        
        success = process_with_garnet(
            output_dirs['garnet_input_img'],
            output_dirs['garnet_input_txt'],
            output_dirs['garnet_output']
        )
        
        if success:
            processed_files = os.listdir(output_dirs['garnet_output'])
            print(f"✓ 文本去除完成，生成 {len(processed_files)} 个文件")
    
    print(f"\n结果保存在:")
    for name, path in output_dirs.items():
        file_count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {name}: {path} ({file_count} 个文件)")

if __name__ == "__main__":
    main()
