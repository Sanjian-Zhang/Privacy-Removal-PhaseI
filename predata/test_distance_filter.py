#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试远景人脸过滤功能
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_face_properties(image_path):
    """分析图片中人脸的属性以验证过滤效果"""
    
    # 模拟人脸检测结果
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    img_area = img_width * img_height
    
    print(f"\n📷 图片: {image_path.name}")
    print(f"  尺寸: {img_width}×{img_height} ({img_area:,} 像素)")
    
    # 配置参数（与更新后的代码一致）
    MIN_FACE_SIZE = 120
    CLOSE_UP_FACE_RATIO = 0.12
    MIN_FACE_AREA = 14400
    MAX_DISTANCE_THRESHOLD = 0.6
    MIN_FACE_RESOLUTION = 150
    
    print(f"\n🔍 过滤标准:")
    print(f"  最小人脸尺寸: {MIN_FACE_SIZE}px")
    print(f"  面积比例阈值: {CLOSE_UP_FACE_RATIO:.1%}")
    print(f"  最小面积: {MIN_FACE_AREA:,}px²")
    print(f"  最小分辨率: {MIN_FACE_RESOLUTION}px")
    print(f"  边缘距离阈值: {MAX_DISTANCE_THRESHOLD:.1%}")
    
    # 模拟一些不同大小的人脸来测试过滤效果
    test_faces = [
        {"name": "远景小脸", "width": 60, "height": 80, "x": 100, "y": 100},
        {"name": "中景人脸", "width": 100, "height": 130, "x": 200, "y": 150},
        {"name": "近景大脸", "width": 180, "height": 220, "x": img_width//2-90, "y": img_height//2-110},
        {"name": "边缘人脸", "width": 140, "height": 160, "x": 10, "y": 10},
        {"name": "清晰主体", "width": 250, "height": 300, "x": img_width//2-125, "y": img_height//2-150},
    ]
    
    print(f"\n📊 测试结果:")
    for face in test_faces:
        x, y, w, h = face["x"], face["y"], face["width"], face["height"]
        face_area = w * h
        
        # 检查各项过滤条件
        size_ok = min(w, h) >= MIN_FACE_SIZE
        area_ok = face_area >= MIN_FACE_AREA
        resolution_ok = max(w, h) >= MIN_FACE_RESOLUTION
        
        # 面积比例
        area_ratio = face_area / img_area
        area_ratio_ok = area_ratio >= CLOSE_UP_FACE_RATIO
        
        # 边缘距离
        face_center_x = x + w/2
        face_center_y = y + h/2
        edge_dist_x = min(face_center_x / img_width, (img_width - face_center_x) / img_width)
        edge_dist_y = min(face_center_y / img_height, (img_height - face_center_y) / img_height)
        min_edge_distance = min(edge_dist_x, edge_dist_y)
        edge_ok = min_edge_distance >= (1 - MAX_DISTANCE_THRESHOLD)
        
        # 尺寸比例
        width_ratio = w / img_width
        height_ratio = h / img_height
        size_ratio = max(width_ratio, height_ratio)
        size_ratio_ok = size_ratio >= 0.15
        
        # 综合判断
        passed = size_ok and area_ok and resolution_ok and area_ratio_ok and edge_ok and size_ratio_ok
        
        status = "✅ 通过" if passed else "❌ 过滤"
        print(f"  {face['name']}: {status}")
        print(f"    尺寸: {w}×{h} | 面积: {face_area:,}px² | 比例: {area_ratio:.3f}")
        print(f"    分辨率: {max(w,h)}px | 边缘距离: {min_edge_distance:.3f} | 尺寸比例: {size_ratio:.3f}")
        if not passed:
            reasons = []
            if not size_ok: reasons.append("尺寸太小")
            if not area_ok: reasons.append("面积不足")
            if not resolution_ok: reasons.append("分辨率低")
            if not area_ratio_ok: reasons.append("面积比例低")
            if not edge_ok: reasons.append("太靠近边缘")
            if not size_ratio_ok: reasons.append("尺寸比例小")
            print(f"    过滤原因: {', '.join(reasons)}")

def main():
    """主函数"""
    print("🔍 远景人脸过滤功能测试")
    print("="*50)
    
    # 查找测试图片
    input_dir = Path("/home/zhiqics/sanjian/predata/output_frames69")
    
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 找几张图片进行测试
    image_files = list(input_dir.glob("*.jpg"))[:3]  # 只测试前3张
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中未找到jpg图片")
        return
    
    for image_path in image_files:
        analyze_face_properties(image_path)
    
    print("\n" + "="*50)
    print("✅ 测试完成！")
    print("\n💡 新的过滤策略说明:")
    print("1. 最小人脸尺寸从80px提高到120px")
    print("2. 面积比例阈值从8%提高到12%")
    print("3. 最小面积从6400提高到14400像素²")
    print("4. 新增最小分辨率要求150px")
    print("5. 新增边缘距离检查，避免边缘远景人脸")
    print("6. 新增尺寸比例检查，人脸至少占图片15%")

if __name__ == "__main__":
    main()
