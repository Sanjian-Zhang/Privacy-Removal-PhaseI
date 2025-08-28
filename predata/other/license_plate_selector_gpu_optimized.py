#!/usr/bin/env python3
"""
GPU优化版车牌检测脚本 - 取消内存限制，全面GPU加速
专门针对车牌检测进行优化，提供更高的检测精度和速度
"""

import os
import cv2
import shutil
import argparse
import gc
import psutil
import torch
import datetime
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def check_gpu_availability():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
            print(f"🚀 GPU {i}: {gpu_name} - 总内存: {gpu_memory:.1f}GB, 可用: {gpu_free:.1f}GB")
        return True
    else:
        print("⚠️ 未检测到可用的GPU，将使用CPU")
        return False

def load_license_plate_model(use_gpu=True, gpu_id=None):
    """加载专用车牌检测模型"""
    # 查找 license_plate_detector.pt 模型
    model_paths = [
        './license_plate_detector.pt',
        './models/license_plate_detector.pt',
        './Automatic-License-Plate-Recognition-using-YOLOv8/license_plate_detector.pt',
        '../Automatic-License-Plate-Recognition-using-YOLOv8/license_plate_detector.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 未找到 license_plate_detector.pt 模型文件")
        print("请确保模型文件在以下位置之一：")
        for path in model_paths:
            print(f"   - {path}")
        raise FileNotFoundError("license_plate_detector.pt not found")
    
    print(f"🎯 加载专用车牌检测模型: {model_path}")
    model = YOLO(model_path)
    
    # 设置设备
    if use_gpu and torch.cuda.is_available():
        if gpu_id is not None:
            device = f'cuda:{gpu_id}'
            print(f"🎯 指定使用GPU {gpu_id}")
        else:
            device = 'cuda'
        
        try:
            model.to(device)
            gpu_name = torch.cuda.get_device_name(device)
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            gpu_used = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"🚀 模型已加载到{device}: {gpu_name}")
            print(f"💾 GPU内存: {gpu_used:.1f}GB / {gpu_memory:.1f}GB")
            
            # GPU预热 - 使用更大的张量进行预热
            print("🔥 GPU预热中...")
            dummy_img = torch.randn(4, 3, 640, 640).to(device)  # 增加批次大小
            with torch.no_grad():
                _ = model(dummy_img, verbose=False)
            del dummy_img
            torch.cuda.empty_cache()
            print("✅ GPU预热完成")
        except Exception as e:
            print(f"⚠️ GPU加载失败，回退到CPU: {e}")
            device = 'cpu'
            model.to(device)
    else:
        device = 'cpu'
        print("💻 使用CPU进行推理")
    
    return model, device

def calculate_sharpness(image_region):
    """计算图像区域的清晰度（使用GPU加速）"""
    if torch.cuda.is_available():
        # GPU加速版本
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        gray_tensor = torch.from_numpy(gray).float().cuda()
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0)
        grad_x = torch.nn.functional.conv2d(gray_tensor, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray_tensor, sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        sharpness = torch.var(gradient_magnitude).item()
        
        del gray_tensor, grad_x, grad_y, gradient_magnitude
        torch.cuda.empty_cache()
        return sharpness
    else:
        # CPU版本
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_contrast(image_region):
    """计算图像区域的对比度（使用GPU加速）"""
    if torch.cuda.is_available():
        # GPU加速版本
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        gray_tensor = torch.from_numpy(gray).float().cuda()
        contrast = torch.std(gray_tensor).item()
        del gray_tensor
        torch.cuda.empty_cache()
        return contrast
    else:
        # CPU版本
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        return gray.std()

def evaluate_plate_quality(img, bbox, min_width=60, min_height=15, min_sharpness=50, min_contrast=15):
    """评估车牌质量"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # 基本尺寸检查
    if width < min_width or height < min_height:
        return False, f"尺寸过小: {width}x{height}"
    
    # 长宽比检查（车牌通常是横向的）
    aspect_ratio = width / height
    if aspect_ratio < 1.5 or aspect_ratio > 8:
        return False, f"长宽比异常: {aspect_ratio:.2f}"
    
    # 提取车牌区域
    plate_region = img[y1:y2, x1:x2]
    if plate_region.size == 0:
        return False, "车牌区域为空"
    
    # 计算清晰度和对比度
    sharpness = calculate_sharpness(plate_region)
    contrast = calculate_contrast(plate_region)
    
    # 质量检查
    if sharpness < min_sharpness:
        return False, f"清晰度不足: {sharpness:.1f} < {min_sharpness}"
    
    if contrast < min_contrast:
        return False, f"对比度不足: {contrast:.1f} < {min_contrast}"
    
    return True, {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "sharpness": sharpness,
        "contrast": contrast,
        "area": width * height
    }

def detect_license_plates_batch(image_paths, model, device, conf_threshold=0.3):
    """批量检测图片中的车牌（GPU优化）"""
    results = []
    
    # 预加载图像
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            valid_paths.append(img_path)
    
    if not images:
        return results
    
    try:
        # 批量推理
        batch_results = model(images, conf=conf_threshold, device=device, verbose=False)
        
        for i, (result, img_path, img) in enumerate(zip(batch_results, valid_paths, images)):
            detected_plates = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标和置信度
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # 确保坐标在图像范围内
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                    
                    detected_plates.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf)
                    })
            
            results.append({
                'path': img_path,
                'image': img,
                'has_plates': len(detected_plates) > 0,
                'plates': detected_plates
            })
    
    except Exception as e:
        print(f"❌ 批量检测失败: {e}")
        # 回退到单张处理
        for img_path, img in zip(valid_paths, images):
            try:
                result = model(img, conf=conf_threshold, device=device, verbose=False)[0]
                detected_plates = []
                
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                        
                        detected_plates.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf)
                        })
                
                results.append({
                    'path': img_path,
                    'image': img,
                    'has_plates': len(detected_plates) > 0,
                    'plates': detected_plates
                })
            except Exception as e2:
                print(f"❌ 单张检测失败 {img_path}: {e2}")
                results.append({
                    'path': img_path,
                    'image': img,
                    'has_plates': False,
                    'plates': []
                })
    
    return results

def process_images(input_dir, output_dir, model, device, conf_threshold=0.3, 
                  min_sharpness=50, min_contrast=15, copy_mode=True, batch_size=64):
    """处理图片目录，挑选有车牌的图片（GPU优化，无内存限制）"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建输出子目录
    selected_dir = output_path / "selected_plates"
    annotated_dir = output_path / "annotated"
    rejected_dir = output_path / "rejected" if not copy_mode else None
    stats_dir = output_path / "stats"
    
    selected_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)
    if rejected_dir:
        rejected_dir.mkdir(exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("❌ 未找到图片文件")
        return
    
    print(f"📁 处理 {len(image_files)} 张图片（批处理大小: {batch_size}）...")
    print(f"🎯 车牌检测置信度阈值: {conf_threshold}")
    print(f"📏 最小清晰度: {min_sharpness}")
    print(f"🔍 最小对比度: {min_contrast}")
    print(f"🚀 GPU优化模式 - 无内存限制")
    print(f"📸 图片保持原始大小和质量 - 无压缩")
    
    selected_count = 0
    rejected_count = 0
    total_plates = 0
    processed_count = 0
    
    # 用于统计的数据
    detection_stats = []
    
    # 分批处理图片（GPU优化）
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        current_batch = i // batch_size + 1
        
        print(f"\n🚀 处理批次 {current_batch}/{total_batches} - GPU加速批处理...")
        
        # 批量检测车牌
        batch_results = detect_license_plates_batch(batch_files, model, device, conf_threshold)
        
        # 处理批量结果
        for result in tqdm(batch_results, desc=f"批次 {current_batch} 质量评估"):
            processed_count += 1
            img_file = result['path']
            img = result['image']
            has_plates = result['has_plates']
            plates_info = result['plates']
            
            if img is None:
                rejected_count += 1
                continue
            
            # 评估检测到的车牌质量
            valid_plates = []
            if has_plates:
                for plate in plates_info:
                    bbox = plate['bbox']
                    confidence = plate['confidence']
                    
                    # 质量评估
                    is_good, quality_result = evaluate_plate_quality(
                        img, bbox, min_sharpness=min_sharpness, min_contrast=min_contrast
                    )
                    
                    if is_good:
                        plate['quality'] = quality_result
                        valid_plates.append(plate)
                        total_plates += 1
            
            # 记录检测统计
            detection_stats.append({
                'file': img_file.name,
                'detected_plates': len(plates_info) if has_plates else 0,
                'valid_plates': len(valid_plates),
                'selected': len(valid_plates) > 0
            })
            
            if valid_plates:
                selected_count += 1
                
                # 复制/移动原图到选中目录，保持原始格式和质量
                if copy_mode:
                    shutil.copy2(img_file, selected_dir / img_file.name)
                else:
                    shutil.move(str(img_file), selected_dir / img_file.name)
                
                # 创建标注图片，保持原始尺寸和质量
                annotated_img = img.copy()
                for plate in valid_plates:
                    x1, y1, x2, y2 = plate['bbox']
                    conf = plate['confidence']
                    quality = plate['quality']
                    
                    # 绘制边界框（根据图片大小调整线条粗细）
                    line_thickness = max(2, min(img.shape[0], img.shape[1]) // 500)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
                    
                    # 根据图片尺寸调整字体大小
                    font_scale = max(0.5, min(img.shape[0], img.shape[1]) / 1000)
                    font_thickness = max(1, int(font_scale * 2))
                    
                    # 添加详细标签
                    label = f"车牌 {conf:.2f}"
                    quality_label = f"清晰度:{quality['sharpness']:.0f} 对比度:{quality['contrast']:.0f}"
                    
                    # 计算文字位置，确保在图片范围内
                    text_y1 = max(30, y1 - 10)
                    text_y2 = max(50, y1 + 10) if text_y1 == 30 else y1 - 30
                    
                    cv2.putText(annotated_img, label, (x1, text_y1), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                    cv2.putText(annotated_img, quality_label, (x1, text_y2), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 0), max(1, font_thickness - 1))
                
                # 保存标注图片，保持原始质量（无压缩）
                # 获取原始图片的扩展名
                original_ext = img_file.suffix.lower()
                
                if original_ext in ['.jpg', '.jpeg']:
                    # 对于JPEG格式，使用最高质量设置
                    cv2.imwrite(str(annotated_dir / img_file.name), annotated_img, 
                               [cv2.IMWRITE_JPEG_QUALITY, 100])
                elif original_ext == '.png':
                    # 对于PNG格式，使用无损压缩
                    cv2.imwrite(str(annotated_dir / img_file.name), annotated_img, 
                               [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    # 其他格式直接保存
                    cv2.imwrite(str(annotated_dir / img_file.name), annotated_img)
                
            else:
                rejected_count += 1
                if rejected_dir and not copy_mode:
                    shutil.move(str(img_file), rejected_dir / img_file.name)
        
        # 显示当前进度和内存使用情况
        memory_usage = psutil.virtual_memory().used / (1024**3)
        gpu_info = ""
        if device != 'cpu':
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            gpu_info = f", GPU: {gpu_memory_used:.1f}GB/{gpu_memory_total:.1f}GB"
        
        print(f"💾 内存使用: {memory_usage:.1f}GB{gpu_info}")
        print(f"📊 当前进度: 已处理 {processed_count}/{len(image_files)} 张，选中 {selected_count} 张")
        print(f"⚡ 当前选中率: {selected_count/processed_count*100:.1f}%")
        
        # 定期清理GPU内存（但不强制限制）
        if device != 'cpu' and current_batch % 5 == 0:
            torch.cuda.empty_cache()
    
    # 保存统计信息
    stats_file = stats_dir / "detection_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("GPU优化车牌检测统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理时间: {datetime.datetime.now()}\n")
        f.write(f"处理模式: GPU优化批处理（无内存限制）\n")
        f.write(f"图片质量: 保持原始大小和质量（无压缩）\n")
        f.write(f"使用设备: {device}\n")
        f.write(f"批处理大小: {batch_size}\n")
        f.write(f"总图片数: {len(image_files)}\n")
        f.write(f"选中图片: {selected_count}\n")
        f.write(f"拒绝图片: {rejected_count}\n")
        f.write(f"检测到车牌总数: {total_plates}\n")
        f.write(f"选中率: {selected_count/len(image_files)*100:.1f}%\n")
        f.write(f"检测置信度阈值: {conf_threshold}\n")
        f.write(f"清晰度阈值: {min_sharpness}\n")
        f.write(f"对比度阈值: {min_contrast}\n")
        f.write("\n详细结果:\n")
        for stat in detection_stats:
            status = "✅" if stat['selected'] else "❌"
            f.write(f"{status} {stat['file']}: 检测{stat['detected_plates']}个，有效{stat['valid_plates']}个\n")
    
    print(f"\n🎉 GPU优化处理完成！")
    print(f"📊 统计信息：")
    print(f"   - 总图片数：{len(image_files)}")
    print(f"   - 有车牌图片：{selected_count}")
    print(f"   - 无车牌图片：{rejected_count}")
    print(f"   - 检测到车牌总数：{total_plates}")
    print(f"   - 选中率：{selected_count/len(image_files)*100:.1f}%")
    print(f"   - 平均每张图片车牌数：{total_plates/selected_count if selected_count > 0 else 0:.1f}")
    print(f"\n📁 输出目录：")
    print(f"   - 选中图片：{selected_dir}")
    print(f"   - 标注图片：{annotated_dir}")
    print(f"   - 统计文件：{stats_file}")
    if rejected_dir:
        print(f"   - 拒绝图片：{rejected_dir}")

def main():
    parser = argparse.ArgumentParser(description="GPU优化车牌检测脚本 - 无内存限制")
    parser.add_argument("--input", "-i", required=True, help="输入图片目录")
    parser.add_argument("--output", "-o", default="./license_plate_results_gpu", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.3, help="车牌检测置信度阈值 (0-1)")
    parser.add_argument("--sharpness", type=float, default=50, help="最小清晰度阈值")
    parser.add_argument("--contrast", type=float, default=15, help="最小对比度阈值")
    parser.add_argument("--copy", action="store_true", help="复制模式（默认移动模式）")
    parser.add_argument("--no_gpu", action="store_true", help="禁用GPU，强制使用CPU")
    parser.add_argument("--gpu_id", type=int, default=None, help="指定使用的GPU ID (0, 1, 2...)")
    parser.add_argument("--batch_size", type=int, default=128, help="GPU优化批处理大小（推荐128-256）")
    
    args = parser.parse_args()
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    use_gpu = gpu_available and not args.no_gpu
    
    # 显示初始内存状态
    initial_memory = psutil.virtual_memory().used / (1024**3)
    print(f"🚀 开始GPU优化处理，当前内存使用: {initial_memory:.1f}GB")
    
    if use_gpu:
        print("⚡ 启用GPU加速模式 - 无内存限制")
        # 设置PyTorch GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("💻 使用CPU模式")
    
    try:
        # 加载专用车牌检测模型
        model, device = load_license_plate_model(use_gpu, args.gpu_id)
        
        # 处理图片
        process_images(
            input_dir=args.input,
            output_dir=args.output,
            model=model,
            device=device,
            conf_threshold=args.conf,
            min_sharpness=args.sharpness,
            min_contrast=args.contrast,
            copy_mode=args.copy,
            batch_size=args.batch_size
        )
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU内存
        if use_gpu:
            torch.cuda.empty_cache()
            print("🧹 GPU内存清理完成")

if __name__ == "__main__":
    main()
