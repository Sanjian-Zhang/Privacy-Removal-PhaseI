#!/usr/bin/env python3
"""
挑选出有清晰车牌的图片
使用 YOLOv8 检测车牌，并根据置信度和清晰度筛选
"""

import os
import cv2
import shutil
import argparse
import gc
import psutil
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def calculate_sharpness(image_region):
    """计算图像区域的清晰度（使用拉普拉斯方差）"""
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_contrast(image_region):
    """计算图像区域的对比度"""
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
    return gray.std()

def calculate_edge_density(image_region):
    """计算边缘密度（用于评估文字清晰度）"""
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density

def calculate_text_clarity(image_region):
    """计算文字清晰度评分"""
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
    
    # 方法1: 梯度幅度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # 方法2: 局部标准差
    kernel = cv2.getGaussianKernel(9, 1.5)
    kernel = kernel @ kernel.T
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_variance = local_sq_mean - local_mean**2
    avg_local_std = np.mean(np.sqrt(np.maximum(local_variance, 0)))
    
    # 综合评分
    text_clarity_score = avg_gradient * 0.6 + avg_local_std * 0.4
    return text_clarity_score

def assess_plate_quality(image_region, min_width=80, min_height=20):
    """综合评估车牌质量"""
    height, width = image_region.shape[:2]
    
    # 尺寸检查
    if width < min_width or height < min_height:
        return False, {"reason": "尺寸过小", "width": width, "height": height}
    
    # 计算各种清晰度指标
    sharpness = calculate_sharpness(image_region)
    contrast = calculate_contrast(image_region)
    edge_density = calculate_edge_density(image_region)
    text_clarity = calculate_text_clarity(image_region)
    
    quality_metrics = {
        "sharpness": sharpness,
        "contrast": contrast,
        "edge_density": edge_density,
        "text_clarity": text_clarity,
        "width": width,
        "height": height
    }
    
    return True, quality_metrics

def is_plate_size_reasonable(bbox, img_shape, min_area=800, max_ratio=0.3, min_width=80, min_height=20,
                           min_area_ratio=0.001, max_area_ratio=0.2):
    """检查车牌尺寸是否合理（提高最小尺寸要求，确保近景拍摄）"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    img_area = img_shape[0] * img_shape[1]
    area_ratio = area / img_area
    
    # 检查最小尺寸（像素）
    if width < min_width or height < min_height:
        return False
    
    # 检查最小面积
    if area < min_area:
        return False
    
    # 检查车牌占图片的比例 - 确保是近景拍摄
    if area_ratio < min_area_ratio:  # 车牌太小，可能是远景
        return False
    
    # 检查车牌不能占据图片太大比例
    if area_ratio > max_area_ratio:  # 车牌太大，可能是特写
        return False
    
    # 检查长宽比是否合理（车牌通常是横向的）
    aspect_ratio = width / height
    if aspect_ratio < 2.0 or aspect_ratio > 6:  # 更严格的长宽比
        return False
    
    return True

def is_near_view_plate(bbox, img_shape, min_width_ratio=0.08, min_height_ratio=0.02, 
                      ideal_y_position=0.4, y_tolerance=0.4):
    """判断车牌是否为近景拍摄"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    img_height, img_width = img_shape[:2]
    
    # 计算车牌相对尺寸
    width_ratio = width / img_width
    height_ratio = height / img_height
    
    # 车牌宽度至少占图片宽度的8%，高度至少占2%
    if width_ratio < min_width_ratio or height_ratio < min_height_ratio:
        return False, f"车牌过小: 宽度比{width_ratio:.3f} < {min_width_ratio}, 高度比{height_ratio:.3f} < {min_height_ratio}"
    
    # 检查车牌在图片中的垂直位置（近景车牌通常在图片中下部）
    plate_center_y = (y1 + y2) / 2
    y_position_ratio = plate_center_y / img_height
    
    # 近景车牌一般在图片的30%-80%高度范围内
    if abs(y_position_ratio - ideal_y_position) > y_tolerance:
        return False, f"车牌位置异常: Y位置比{y_position_ratio:.3f}偏离理想位置{ideal_y_position}超过{y_tolerance}"
    
    # 计算近景评分
    near_score = (
        min(width_ratio / 0.15, 1.0) * 0.4 +  # 宽度评分
        min(height_ratio / 0.05, 1.0) * 0.3 +  # 高度评分
        (1 - abs(y_position_ratio - ideal_y_position) / y_tolerance) * 0.3  # 位置评分
    )
    
    return True, {
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "y_position_ratio": y_position_ratio,
        "near_score": near_score
    }

def check_memory_usage(threshold_gb=100):
    """检查内存使用情况，如果超过阈值则强制垃圾回收"""
    memory_gb = psutil.virtual_memory().used / (1024**3)
    if memory_gb > threshold_gb:
        print(f"⚠️ 内存使用量过高: {memory_gb:.1f}GB，执行垃圾回收...")
        gc.collect()
        return True
    return False

def detect_clear_plates(image_path, model, conf_threshold=0.5, sharpness_threshold=150, 
                       contrast_threshold=25, edge_density_threshold=0.05, text_clarity_threshold=20,
                       require_two_plates=True):
    """检测图片中的清晰车牌（使用更严格的标准）"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False, []
    
    try:
        # YOLO 检测
        results = model(img, conf=conf_threshold, verbose=False)
        
        clear_plates = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # 确保坐标在图像范围内
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                    
                    # 检查车牌尺寸是否合理
                    if not is_plate_size_reasonable([x1, y1, x2, y2], img.shape):
                        continue
                    
                    # 检查是否为近景车牌
                    is_near, near_info = is_near_view_plate([x1, y1, x2, y2], img.shape)
                    if not is_near:
                        continue  # 跳过远景车牌
                    
                    # 提取车牌区域
                    plate_region = img[y1:y2, x1:x2]
                    if plate_region.size == 0:
                        continue
                    
                    # 综合质量评估
                    is_valid, quality_metrics = assess_plate_quality(plate_region)
                    if not is_valid:
                        continue
                    
                    # 使用更严格的阈值判断
                    sharpness = quality_metrics['sharpness']
                    contrast = quality_metrics['contrast']
                    edge_density = quality_metrics['edge_density']
                    text_clarity = quality_metrics['text_clarity']
                    
                    # 多重质量检查
                    quality_checks = [
                        sharpness >= sharpness_threshold,
                        contrast >= contrast_threshold,
                        edge_density >= edge_density_threshold,
                        text_clarity >= text_clarity_threshold
                    ]
                    
                    # 至少要满足3个条件才认为是清晰车牌
                    if sum(quality_checks) >= 3:
                        # 计算综合质量评分（包含近景评分）
                        near_score = near_info.get('near_score', 0) if isinstance(near_info, dict) else 0
                        quality_score = (
                            (sharpness / 200) * 0.25 +
                            (contrast / 50) * 0.2 +
                            (edge_density / 0.1) * 0.2 +
                            (text_clarity / 30) * 0.15 +
                            near_score * 0.2  # 近景评分权重
                        )
                        
                        clear_plates.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'sharpness': sharpness,
                            'contrast': contrast,
                            'edge_density': edge_density,
                            'text_clarity': text_clarity,
                            'quality_score': quality_score,
                            'near_score': near_score,
                            'near_info': near_info
                        })
        
        # 如果需要2个车牌才满足要求
        if require_two_plates:
            meets_requirement = len(clear_plates) >= 2
        else:
            meets_requirement = len(clear_plates) > 0
        
        return meets_requirement, clear_plates
    
    finally:
        # 释放图像内存
        del img
        gc.collect()

def process_images(input_dir, output_dir, model_path=None, conf_threshold=0.5, 
                  sharpness_threshold=150, contrast_threshold=25, copy_original=True, batch_size=50,
                  edge_density_threshold=0.05, text_clarity_threshold=20, require_two_plates=True):
    """处理图片目录，挑选出有清晰车牌的图片"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    if require_two_plates:
        selected_dir = output_path / "two_plates_selected"
        annotated_dir = output_path / "two_plates_annotated" 
        rejected_dir = output_path / "not_two_plates" if not copy_original else None
    else:
        selected_dir = output_path / "selected_plates"
        annotated_dir = output_path / "annotated"
        rejected_dir = output_path / "rejected" if not copy_original else None
    
    selected_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)
    if rejected_dir:
        rejected_dir.mkdir(exist_ok=True)
    
    # 加载 YOLO 模型
    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    else:
        print("⚠️ 使用预训练的 YOLOv8n 模型（可能需要下载）")
        model = YOLO('yolov8n.pt')  # 会自动下载预训练模型
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("❌ 未找到图片文件")
        return
    
    plate_requirement = "至少2个" if require_two_plates else "至少1个"
    print(f"📁 处理 {len(image_files)} 张图片，查找有{plate_requirement}清晰车牌的图片（批处理大小: {batch_size}）...")
    
    selected_count = 0
    total_plates = 0
    processed_count = 0
    
    # 分批处理图片
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        print(f"\n🔄 处理批次 {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        for img_file in tqdm(batch_files, desc=f"批次 {i//batch_size + 1}"):
            processed_count += 1
            
            # 检查内存使用情况
            if processed_count % 10 == 0:
                check_memory_usage(100)  # 100GB阈值
            
            has_clear_plates, plates_info = detect_clear_plates(
                img_file, model, conf_threshold, sharpness_threshold, contrast_threshold,
                edge_density_threshold, text_clarity_threshold, require_two_plates
            )
            
            if has_clear_plates:
                selected_count += 1
                total_plates += len(plates_info)
                
                # 移动原图到选中目录
                if copy_original:
                    shutil.copy2(str(img_file), selected_dir / img_file.name)
                else:
                    shutil.move(str(img_file), selected_dir / img_file.name)
                
                # 创建标注图片
                img = cv2.imread(str(img_file))
                if img is not None:
                    try:
                        for plate in plates_info:
                            x1, y1, x2, y2 = plate['bbox']
                            conf = plate['confidence']
                            
                            # 绘制边界框
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # 添加信息标签（包含质量评分和近景评分）
                            quality_score = plate.get('quality_score', 0)
                            near_score = plate.get('near_score', 0)
                            near_info = plate.get('near_info', {})
                            width_ratio = near_info.get('width_ratio', 0) if isinstance(near_info, dict) else 0
                            
                            label = f"Q:{quality_score:.2f} N:{near_score:.2f} W:{width_ratio:.3f}"
                            cv2.putText(img, label, (int(x1), int(y1-10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # 在图片右上角添加车牌数量标记
                        plate_count_label = f"Plates: {len(plates_info)}"
                        cv2.putText(img, plate_count_label, (img.shape[1] - 150, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # 保存标注图片
                        cv2.imwrite(str(annotated_dir / img_file.name), img)
                    finally:
                        del img  # 释放图像内存
            
            elif rejected_dir:
                # 移动未选中的图片到拒绝目录
                shutil.move(str(img_file), rejected_dir / img_file.name)
        
        # 每批处理完后强制垃圾回收
        gc.collect()
        
        # 显示当前进度
        memory_usage = psutil.virtual_memory().used / (1024**3)
        print(f"💾 当前内存使用: {memory_usage:.1f}GB / 120GB")
        print(f"📊 当前进度: 已处理 {processed_count}/{len(image_files)} 张图片，选中 {selected_count} 张")
    
    print(f"\n✅ 处理完成！")
    print(f"📊 统计信息：")
    print(f"   - 总图片数：{len(image_files)}")
    print(f"   - 符合要求的图片：{selected_count}（有{plate_requirement}清晰车牌）")
    print(f"   - 检测到车牌总数：{total_plates}")
    print(f"   - 选中率：{selected_count/len(image_files)*100:.1f}%")
    print(f"\n📁 输出目录：")
    print(f"   - 选中图片：{selected_dir}")
    print(f"   - 标注图片：{annotated_dir}")
    if rejected_dir:
        print(f"   - 拒绝图片：{rejected_dir}")

def main():
    parser = argparse.ArgumentParser(description="挑选有清晰车牌的图片")
    parser.add_argument("--input", "-i", required=True, help="输入图片目录")
    parser.add_argument("--output", "-o", default="./plate_selection", help="输出目录")
    parser.add_argument("--model", help="自定义 YOLO 模型路径（可选）")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值 (0-1)")
    parser.add_argument("--sharpness", type=float, default=150, help="清晰度阈值（更高=更严格）")
    parser.add_argument("--contrast", type=float, default=25, help="对比度阈值（更高=更严格）")
    parser.add_argument("--edge_density", type=float, default=0.05, help="边缘密度阈值")
    parser.add_argument("--text_clarity", type=float, default=20, help="文字清晰度阈值")
    parser.add_argument("--min_width_ratio", type=float, default=0.08, help="车牌最小宽度比例（近景要求）")
    parser.add_argument("--min_height_ratio", type=float, default=0.02, help="车牌最小高度比例（近景要求）")
    parser.add_argument("--copy", action="store_true", help="复制原图到选中目录（默认移动）")
    parser.add_argument("--batch_size", type=int, default=50, help="批处理大小（避免内存不足）")
    parser.add_argument("--memory_limit", type=int, default=100, help="内存使用限制（GB）")
    parser.add_argument("--require_two_plates", action="store_true", default=True, 
                       help="是否要求至少2个清晰车牌才选中图片（默认True）")
    parser.add_argument("--single_plate_ok", action="store_true", 
                       help="允许单个车牌的图片（将require_two_plates设为False）")
    
    args = parser.parse_args()
    
    # 处理车牌数量要求
    if args.single_plate_ok:
        require_two_plates = False
        requirement_desc = "至少1个"
    else:
        require_two_plates = args.require_two_plates
        requirement_desc = "至少2个" if require_two_plates else "至少1个"
    
    print(f"🎯 设置：只选择有{requirement_desc}清晰车牌的图片")
    
    # 检查初始内存状态
    initial_memory = psutil.virtual_memory().used / (1024**3)
    print(f"🚀 开始处理，当前内存使用: {initial_memory:.1f}GB / 120GB")
    
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        sharpness_threshold=args.sharpness,
        contrast_threshold=args.contrast,
        copy_original=args.copy,
        batch_size=args.batch_size,
        edge_density_threshold=args.edge_density,
        text_clarity_threshold=args.text_clarity,
        require_two_plates=require_two_plates
    )

if __name__ == "__main__":
    main()
