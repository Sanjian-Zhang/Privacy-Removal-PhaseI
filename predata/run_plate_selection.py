#!/usr/bin/env python3
"""
运行车牌选择脚本的示例
选择有2个清晰车牌的图片，将它们移动到对应的文件夹
"""

import subprocess
import sys
import os
from pathlib import Path

def run_plate_selection(input_dir, output_dir, require_two_plates=True, copy_files=True):
    """
    运行车牌选择脚本
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        require_two_plates: 是否要求至少2个车牌
        copy_files: 是否复制文件（True）还是移动文件（False）
    """
    
    # 构建命令
    cmd = [
        sys.executable, 
        "select_clear_plates.py",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--conf", "0.4",  # 降低置信度阈值以检测更多车牌
        "--sharpness", "100",  # 降低清晰度要求
        "--contrast", "20",    # 降低对比度要求
        "--batch_size", "20"   # 较小的批次大小
    ]
    
    # 添加车牌数量要求参数
    if require_two_plates:
        cmd.append("--require_two_plates")
    else:
        cmd.append("--single_plate_ok")
    
    # 添加文件操作模式
    if copy_files:
        cmd.append("--copy")
    
    print(f"🚀 执行命令: {' '.join(cmd)}")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 车牌要求: {'至少2个' if require_two_plates else '至少1个'}")
    print(f"📋 文件操作: {'复制' if copy_files else '移动'}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("\n" + "="*50)
        print("🔍 程序输出:")
        print("="*50)
        print(result.stdout)
        
        if result.stderr:
            print("\n" + "="*50)
            print("⚠️ 错误信息:")
            print("="*50)
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ 执行成功！返回码: {result.returncode}")
        else:
            print(f"\n❌ 执行失败！返回码: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 执行时发生错误: {e}")
        return False

def main():
    # 设置输入和输出目录
    current_dir = Path(__file__).parent
    
    # 可以选择不同的输入目录进行测试
    test_options = [
        ("test_images", "测试图片目录"),
        ("output_frames70/1-2_faces", "有1-2个人脸的图片"),
        ("output_frames70/3-6_faces", "有3-6个人脸的图片"),
        ("classified_images70", "已分类的图片"),
    ]
    
    print("📋 可用的测试目录:")
    for i, (dir_name, desc) in enumerate(test_options, 1):
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"  {i}. {dir_name} - {desc}")
        else:
            print(f"  {i}. {dir_name} - {desc} (不存在)")
    
    # 选择一个存在的目录进行测试
    for dir_name, desc in test_options:
        input_dir = current_dir / dir_name
        if input_dir.exists():
            print(f"\n🎯 选择测试目录: {dir_name}")
            
            # 设置输出目录
            output_dir = current_dir / f"plate_selection_results_{dir_name.replace('/', '_')}"
            
            print(f"\n📋 测试场景1: 查找有2个清晰车牌的图片")
            success1 = run_plate_selection(
                input_dir=input_dir,
                output_dir=output_dir / "two_plates_test",
                require_two_plates=True,
                copy_files=True  # 复制文件而不是移动
            )
            
            print(f"\n📋 测试场景2: 查找有至少1个清晰车牌的图片")
            success2 = run_plate_selection(
                input_dir=input_dir,
                output_dir=output_dir / "single_plate_test", 
                require_two_plates=False,
                copy_files=True  # 复制文件而不是移动
            )
            
            if success1 or success2:
                print(f"\n📁 结果保存在: {output_dir}")
                print("📋 查看结果目录结构:")
                if output_dir.exists():
                    for item in output_dir.rglob("*"):
                        if item.is_dir():
                            print(f"  📁 {item.relative_to(output_dir)}/")
                        else:
                            print(f"  📄 {item.relative_to(output_dir)}")
            
            break
    else:
        print("\n❌ 没有找到可用的测试目录")

if __name__ == "__main__":
    main()
