#!/usr/bin/env python3
"""
演示如何结合COCO标注标签使用Garnet模型处理图片
这是一个简化的示例脚本，展示了主要处理流程
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def load_coco_data(annotation_file):
    """加载COCO标注数据"""
    with open(annotation_file, 'r') as f:
        data = json.loads(f.read())
    
    # 建立类别映射
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 建立图片索引
    images = {img['id']: img for img in data['images']}
    
    # 按图片分组标注
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    return categories, images, annotations_by_image

def filter_text_annotations(annotations, text_category_id=3):
    """筛选出文本类别的标注"""
    return [ann for ann in annotations if ann['category_id'] == text_category_id]

def create_garnet_input(image, text_annotations, output_txt_path, output_img_path):
    """为Garnet模型创建输入文件"""
    height, width = image.shape[:2]
    
    # 创建文本区域定义文件
    with open(output_txt_path, 'w') as f:
        for i, ann in enumerate(text_annotations):
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 转换为Garnet需要的格式
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            f.write(f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2}\n")
    
    # 保存图片
    cv2.imwrite(output_img_path, image)
    print(f"创建Garnet输入: {output_txt_path}, {output_img_path}")

def visualize_annotations(image, annotations, categories, output_path):
    """可视化标注"""
    vis_image = image.copy()
    
    colors = {
        1: (0, 255, 0),    # 人脸/车辆 - 绿色
        2: (255, 0, 0),    # 物体 - 蓝色  
        3: (0, 0, 255),    # 文本/车牌 - 红色
    }
    
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        category_id = ann['category_id']
        category_name = categories.get(category_id, f"未知类别{category_id}")
        
        color = colors.get(category_id, (128, 128, 128))
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # 添加标签
        label = f"{category_name} (ID:{category_id})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis_image)
    print(f"可视化保存到: {output_path}")

def process_single_image(image_path, image_id, annotations, categories, output_dir):
    """处理单张图片"""
    print(f"\n=== 处理图片: {os.path.basename(image_path)} ===")
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    print(f"图片尺寸: {image.shape}")
    print(f"标注数量: {len(annotations)}")
    
    # 分析标注
    category_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f"未知{cat_id}")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    print("类别分布:", category_counts)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 生成可视化
    vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    visualize_annotations(image, annotations, categories, vis_path)
    
    # 2. 筛选文本标注并创建Garnet输入
    text_annotations = filter_text_annotations(annotations, text_category_id=3)
    if text_annotations:
        print(f"发现 {len(text_annotations)} 个文本区域")
        
        # 创建Garnet输入文件
        txt_path = os.path.join(output_dir, f"{base_name}_text_regions.txt")
        img_path = os.path.join(output_dir, f"{base_name}_garnet_input.jpg")
        create_garnet_input(image, text_annotations, txt_path, img_path)
        
        print("Garnet处理命令示例:")
        print(f"python garnet_inference.py --input_image {img_path} --input_txt {txt_path} --output_dir {output_dir}")
    else:
        print("未发现文本区域")

def main():
    parser = argparse.ArgumentParser(description="COCO标注+Garnet模型演示")
    parser.add_argument("--images_dir", default="/home/zhiqics/sanjian/dataset/images/Train",
                       help="图片目录路径")
    parser.add_argument("--annotations", default="/home/zhiqics/sanjian/dataset/annotations/instances_Train.json",
                       help="COCO标注文件路径")
    parser.add_argument("--output_dir", default="/home/zhiqics/sanjian/demo_output",
                       help="输出目录")
    parser.add_argument("--image_name", default="frame_00030.jpg",
                       help="要处理的图片名称")
    
    args = parser.parse_args()
    
    print("=== COCO标注 + Garnet模型处理演示 ===")
    print(f"图片目录: {args.images_dir}")
    print(f"标注文件: {args.annotations}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理图片: {args.image_name}")
    
    # 加载COCO数据
    print("\n加载COCO标注数据...")
    categories, images, annotations_by_image = load_coco_data(args.annotations)
    
    print(f"类别定义: {categories}")
    print(f"总图片数: {len(images)}")
    print(f"总标注数: {sum(len(anns) for anns in annotations_by_image.values())}")
    
    # 查找指定图片
    target_image_id = None
    for img_id, img_info in images.items():
        if img_info['file_name'] == args.image_name:
            target_image_id = img_id
            break
    
    if target_image_id is None:
        print(f"未找到图片: {args.image_name}")
        print("可用图片示例:")
        for i, (img_id, img_info) in enumerate(list(images.items())[:5]):
            print(f"  {img_info['file_name']}")
        return
    
    # 获取图片标注
    image_annotations = annotations_by_image.get(target_image_id, [])
    image_path = os.path.join(args.images_dir, args.image_name)
    
    # 处理图片
    process_single_image(image_path, target_image_id, image_annotations, categories, args.output_dir)
    
    print(f"\n处理完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
