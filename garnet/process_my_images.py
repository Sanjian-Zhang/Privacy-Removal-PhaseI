#!/usr/bin/env python3
"""
处理自己图片的便捷脚本
支持自动文本检测或使用提供的坐标文件
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path

# 添加CODE目录到路径
sys.path.append('./CODE')

def setup_directories(input_dir, output_dir, box_dir=None):
    """创建必要的目录结构"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if box_dir:
        Path(box_dir).mkdir(parents=True, exist_ok=True)
    return True

def detect_text_with_paddleocr(image_path, output_txt_path):
    """使用PaddleOCR检测文本区域并保存坐标"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        result = ocr.ocr(image_path, cls=True)
        
        with open(output_txt_path, 'w') as f:
            if result[0] is not None:
                for line in result[0]:
                    box = line[0]  # 获取边界框坐标
                    # 转换为所需格式：x1,y1,x2,y2,x3,y3,x4,y4
                    coords = []
                    for point in box:
                        coords.extend([int(point[0]), int(point[1])])
                    f.write(','.join(map(str, coords)) + '\\n')
        
        return True
    except ImportError:
        print("❌ PaddleOCR未安装。请运行: pip install paddleocr")
        return False
    except Exception as e:
        print(f"❌ PaddleOCR检测失败: {e}")
        return False

def detect_text_with_opencv(image_path, output_txt_path):
    """使用OpenCV简单文本检测（备用方案）"""
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用形态学操作检测文本区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # 找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(output_txt_path, 'w') as f:
            for contour in contours:
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:  # 过滤太小的区域
                    # 转换为四个角点
                    coords = [x, y, x+w, y, x+w, y+h, x, y+h]
                    f.write(','.join(map(str, coords)) + '\\n')
        
        return True
    except Exception as e:
        print(f"❌ OpenCV检测失败: {e}")
        return False

def validate_data(image_dir, box_dir):
    """验证图片和坐标文件的对应关系"""
    image_files = set(Path(f).stem for f in glob.glob(os.path.join(image_dir, "*.jpg")))
    box_files = set(Path(f).stem for f in glob.glob(os.path.join(box_dir, "*.txt")))
    
    missing_boxes = image_files - box_files
    missing_images = box_files - image_files
    
    if missing_boxes:
        print(f"⚠️  缺少坐标文件: {missing_boxes}")
    if missing_images:
        print(f"⚠️  缺少图片文件: {missing_images}")
    
    valid_pairs = image_files & box_files
    print(f"✓ 找到 {len(valid_pairs)} 对有效的图片-坐标文件")
    
    return len(valid_pairs) > 0

def run_garnet_inference(image_dir, box_dir, output_dir, gpu=True):
    """运行GaRNet推理"""
    cmd = f"""
    python CODE/inference.py \\
        --image_path {image_dir} \\
        --box_path {box_dir} \\
        --result_path {output_dir} \\
        --input_size 512 \\
        --model_path ./WEIGHTS/GaRNet/saved_model.pth
    """
    
    if gpu:
        cmd += " --gpu"
    
    print(f"🚀 运行GaRNet推理...")
    print(f"命令: {cmd}")
    
    return os.system(cmd) == 0

def main():
    parser = argparse.ArgumentParser(description="处理自己的图片 - GaRNet文本移除")
    parser.add_argument("--input_dir", required=True, help="输入图片目录")
    parser.add_argument("--output_dir", required=True, help="输出结果目录")
    parser.add_argument("--box_dir", help="坐标文件目录（如果不提供将自动检测）")
    parser.add_argument("--auto_detect", choices=["paddleocr", "opencv"], 
                       default="paddleocr", help="自动文本检测方法")
    parser.add_argument("--gpu", action="store_true", help="使用GPU")
    parser.add_argument("--validate_only", action="store_true", help="仅验证数据格式")
    
    args = parser.parse_args()
    
    print("🎯 GaRNet 图片处理工具")
    print("=" * 50)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 查找图片文件
    image_files = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_files:
        print(f"❌ 在 {args.input_dir} 中没有找到.jpg文件")
        return
    
    print(f"✓ 找到 {len(image_files)} 个图片文件")
    
    # 设置坐标文件目录
    if not args.box_dir:
        args.box_dir = os.path.join(os.path.dirname(args.input_dir), "auto_detected_boxes")
        print(f"📁 将自动检测文本并保存到: {args.box_dir}")
    
    # 创建目录
    setup_directories(args.input_dir, args.output_dir, args.box_dir)
    
    # 自动检测文本（如果需要）
    if not args.box_dir or not os.path.exists(args.box_dir) or len(glob.glob(os.path.join(args.box_dir, "*.txt"))) == 0:
        print(f"🔍 使用 {args.auto_detect} 自动检测文本区域...")
        
        for image_file in image_files:
            image_name = Path(image_file).stem
            txt_file = os.path.join(args.box_dir, f"{image_name}.txt")
            
            print(f"   处理: {image_name}")
            
            if args.auto_detect == "paddleocr":
                success = detect_text_with_paddleocr(image_file, txt_file)
            else:
                success = detect_text_with_opencv(image_file, txt_file)
            
            if not success:
                # 创建空文件作为备用
                with open(txt_file, 'w') as f:
                    pass
                print(f"   ⚠️  为 {image_name} 创建了空坐标文件")
    
    # 验证数据
    print("\\n🔍 验证数据格式...")
    if not validate_data(args.input_dir, args.box_dir):
        print("❌ 数据验证失败")
        return
    
    if args.validate_only:
        print("✓ 数据验证完成")
        return
    
    # 检查模型文件
    model_path = "./WEIGHTS/GaRNet/saved_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 python download_model.py 下载模型")
        return
    
    # 运行推理
    print("\\n🚀 开始GaRNet文本移除...")
    success = run_garnet_inference(args.input_dir, args.box_dir, args.output_dir, args.gpu)
    
    if success:
        print(f"\\n✅ 处理完成！结果保存在: {args.output_dir}")
        result_files = glob.glob(os.path.join(args.output_dir, "*.png"))
        print(f"📊 生成了 {len(result_files)} 个结果文件")
    else:
        print("\\n❌ 处理失败")

if __name__ == "__main__":
    main()
