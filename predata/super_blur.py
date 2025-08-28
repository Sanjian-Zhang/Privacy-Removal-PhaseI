#!/usr/bin/env python3
import cv2
import os
from pathlib import Path

# 目录设置
input_dir = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images"
output_dir = "/home/zhiqics/sanjian/predata/output_frames70/high_score_images_super_blurred"

# 超强模糊参数
blur_ratio = 0.25    # 右下角1/4区域
blur_strength = 101  # 超强模糊

# 创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"开始处理，模糊强度: {blur_strength}")
count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 读取图片
        img = cv2.imread(input_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        start_x = int(w * (1 - blur_ratio))
        start_y = int(h * (1 - blur_ratio))
        
        # 提取右下角并模糊
        corner = img[start_y:h, start_x:w]
        blurred_corner = cv2.GaussianBlur(corner, (blur_strength, blur_strength), 0)
        
        # 替换原图右下角
        result = img.copy()
        result[start_y:h, start_x:w] = blurred_corner
        
        # 保存
        cv2.imwrite(output_path, result)
        count += 1
        
        if count % 50 == 0:
            print(f"已处理 {count} 张图片...")

print(f"处理完成！共处理 {count} 张图片")
print(f"输出目录: {output_dir}")
