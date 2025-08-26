#!/usr/bin/env python3
"""
验证图片和坐标文件格式的脚本
"""

import os
import argparse
import glob
from pathlib import Path

def validate_txt_format(txt_path):
    """验证txt文件格式"""
    errors = []
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            coords = line.split(',')
            if len(coords) != 8:
                errors.append(f"行 {i}: 应该有8个坐标，实际有{len(coords)}个")
                continue
            
            try:
                coords = [int(x) for x in coords]
            except ValueError:
                errors.append(f"行 {i}: 坐标必须是整数")
                continue
            
            # 检查坐标是否为正数
            if any(x < 0 for x in coords):
                errors.append(f"行 {i}: 坐标不能为负数")
    
    except Exception as e:
        errors.append(f"读取文件失败: {e}")
    
    return errors

def validate_image_file(image_path):
    """验证图片文件"""
    errors = []
    
    if not os.path.exists(image_path):
        errors.append("文件不存在")
        return errors
    
    # 检查文件大小
    size = os.path.getsize(image_path)
    if size == 0:
        errors.append("文件为空")
    elif size < 1024:  # 小于1KB
        errors.append("文件可能损坏（太小）")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="验证GaRNet输入数据格式")
    parser.add_argument("--image_dir", required=True, help="图片目录")
    parser.add_argument("--box_dir", required=True, help="坐标文件目录")
    
    args = parser.parse_args()
    
    print("🔍 验证GaRNet输入数据格式")
    print("=" * 50)
    
    # 检查目录存在性
    if not os.path.exists(args.image_dir):
        print(f"❌ 图片目录不存在: {args.image_dir}")
        return
    
    if not os.path.exists(args.box_dir):
        print(f"❌ 坐标目录不存在: {args.box_dir}")
        return
    
    # 获取文件列表
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    txt_files = glob.glob(os.path.join(args.box_dir, "*.txt"))
    
    print(f"📁 图片文件: {len(image_files)} 个")
    print(f"📁 坐标文件: {len(txt_files)} 个")
    
    # 检查文件名对应关系
    image_names = {Path(f).stem for f in image_files}
    txt_names = {Path(f).stem for f in txt_files}
    
    missing_txt = image_names - txt_names
    missing_img = txt_names - image_names
    matched = image_names & txt_names
    
    print(f"\\n📊 匹配统计:")
    print(f"   ✓ 完全匹配: {len(matched)} 对")
    if missing_txt:
        print(f"   ⚠️  缺少坐标文件: {len(missing_txt)} 个")
        for name in sorted(missing_txt):
            print(f"      - {name}.txt")
    
    if missing_img:
        print(f"   ⚠️  缺少图片文件: {len(missing_img)} 个")
        for name in sorted(missing_img):
            print(f"      - {name}.jpg")
    
    # 详细验证每个文件
    print(f"\\n🔍 详细验证:")
    total_errors = 0
    
    for name in sorted(matched):
        image_path = os.path.join(args.image_dir, f"{name}.jpg")
        txt_path = os.path.join(args.box_dir, f"{name}.txt")
        
        print(f"\\n📄 {name}:")
        
        # 验证图片
        img_errors = validate_image_file(image_path)
        if img_errors:
            print(f"   ❌ 图片问题:")
            for error in img_errors:
                print(f"      - {error}")
            total_errors += len(img_errors)
        else:
            print(f"   ✓ 图片格式正确")
        
        # 验证坐标文件
        txt_errors = validate_txt_format(txt_path)
        if txt_errors:
            print(f"   ❌ 坐标文件问题:")
            for error in txt_errors:
                print(f"      - {error}")
            total_errors += len(txt_errors)
        else:
            # 统计文本区域数量
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            print(f"   ✓ 坐标格式正确 ({len(lines)} 个文本区域)")
    
    # 总结
    print(f"\\n📋 验证总结:")
    print(f"   总文件对数: {len(matched)}")
    print(f"   错误数量: {total_errors}")
    
    if total_errors == 0 and len(matched) > 0:
        print(f"   🎉 所有文件格式正确，可以开始处理！")
        print(f"\\n💡 运行命令:")
        print(f"   python process_my_images.py --input_dir {args.image_dir} --box_dir {args.box_dir} --output_dir ./results")
    elif total_errors > 0:
        print(f"   ⚠️  请修复上述错误后重新验证")
    else:
        print(f"   ❌ 没有找到匹配的文件对")

if __name__ == "__main__":
    main()
