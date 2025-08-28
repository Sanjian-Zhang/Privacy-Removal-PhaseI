#!/usr/bin/env python3
"""
简单的人脸模糊匿名化脚本 - 使用OpenCV
当DeepPrivacy模型下载失败时的替代方案
"""
import os
import cv2
import argparse
import glob
from pathlib import Path

def blur_faces(image_path, output_path, blur_strength=20):
    """
    使用OpenCV检测并模糊人脸
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        blur_strength: 模糊强度
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # 加载人脸检测器
    face_cascade_path = '/home/zhiqics/sanjian/predata/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return False
    
    # 转换为灰度图进行检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Detected {len(faces)} faces")
    
    # 对每个检测到的人脸进行模糊处理
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_region = image[y:y+h, x:x+w]
        
        # 应用高斯模糊
        blurred_face = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
        
        # 将模糊后的人脸放回原图
        image[y:y+h, x:x+w] = blurred_face
    
    # 保存结果
    cv2.imwrite(output_path, image)
    return True

def batch_blur_faces(input_dir, output_dir, blur_strength=20):
    """
    批量处理图片中的人脸模糊
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        blur_strength: 模糊强度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(input_dir, ext)))
        img_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(img_files)} images to process")
    
    success_count = 0
    # 处理每个图片
    for i, img_path in enumerate(img_files):
        try:
            print(f"Processing {i+1}/{len(img_files)}: {os.path.basename(img_path)}")
            
            # 获取输出文件名
            img_name = os.path.basename(img_path)
            name, ext = os.path.splitext(img_name)
            output_path = os.path.join(output_dir, f"{name}_blurred{ext}")
            
            # 模糊人脸
            if blur_faces(img_path, output_path, blur_strength):
                print(f"Saved to: {output_path}")
                success_count += 1
            else:
                print(f"Failed to process: {img_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Batch processing completed! Successfully processed {success_count}/{len(img_files)} images")

def main():
    parser = argparse.ArgumentParser(description='Batch blur faces in images using OpenCV')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed images')
    parser.add_argument('--blur', '-b', type=int, default=20, 
                       help='Blur strength (higher = more blur, must be odd number)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return
    
    # 确保模糊强度是奇数
    if args.blur % 2 == 0:
        args.blur += 1
    
    batch_blur_faces(args.input, args.output, args.blur)

if __name__ == "__main__":
    main()
