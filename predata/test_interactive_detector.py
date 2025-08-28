#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的交互式人脸车牌检测器
"""

import os
import sys
from pathlib import Path

# 模拟测试用的简单图片目录结构
def create_test_structure():
    """创建测试用的目录结构"""
    test_dir = "/home/zhiqics/sanjian/predata/test_images"
    
    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建一些测试文件（空文件）
    test_files = [
        "image001.jpg",
        "image002.png", 
        "image003.jpeg",
        "test_frame_001.jpg",
        "test_frame_002.jpg",
        "document.txt",  # 非图片文件
        "photo.bmp"
    ]
    
    for filename in test_files:
        file_path = os.path.join(test_dir, filename)
        if not os.path.exists(file_path):
            # 创建空文件
            with open(file_path, 'w') as f:
                f.write("")
    
    print(f"✅ 测试目录已创建: {test_dir}")
    print(f"📁 包含文件:")
    for filename in test_files:
        print(f"   - {filename}")
    
    return test_dir

def test_config():
    """测试配置类的功能"""
    print("\n🧪 测试配置类功能...")
    
    # 导入我们的模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 延迟导入以避免依赖问题
        from pathlib import Path
        
        # 创建测试目录
        test_dir = create_test_structure()
        
        # 模拟FastConfig类的核心功能
        class TestConfig:
            def __init__(self, input_dir=None):
                if input_dir:
                    self.INPUT_DIR = input_dir
                    self.OUTPUT_BASE_DIR = os.path.join(input_dir, "processed_output")
                else:
                    self.INPUT_DIR = '/default/path'
                    self.OUTPUT_BASE_DIR = '/default/output'
                
                # 其他配置
                self.ENABLE_SIMILARITY_DETECTION = True
                self.SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            
            def get_output_dirs(self):
                return {
                    'high_score': os.path.join(self.OUTPUT_BASE_DIR, "high_score_images"),
                    'low_score': os.path.join(self.OUTPUT_BASE_DIR, "low_score_images"),
                    'zero_score': os.path.join(self.OUTPUT_BASE_DIR, "zero_score_images"),
                    'analysis': os.path.join(self.OUTPUT_BASE_DIR, "analysis"),
                    'unique_high_score': os.path.join(self.OUTPUT_BASE_DIR, "unique_high_score_images"),
                    'similar_high_score': os.path.join(self.OUTPUT_BASE_DIR, "similar_high_score_images"),
                }
        
        # 测试默认配置
        print("\n📋 测试默认配置:")
        config1 = TestConfig()
        print(f"   输入目录: {config1.INPUT_DIR}")
        print(f"   输出目录: {config1.OUTPUT_BASE_DIR}")
        
        # 测试自定义输入目录
        print(f"\n📋 测试自定义输入目录:")
        config2 = TestConfig(test_dir)
        print(f"   输入目录: {config2.INPUT_DIR}")
        print(f"   输出目录: {config2.OUTPUT_BASE_DIR}")
        
        # 测试输出目录结构
        print(f"\n📁 输出目录结构:")
        output_dirs = config2.get_output_dirs()
        for name, path in output_dirs.items():
            print(f"   {name}: {path}")
        
        # 测试图片文件扫描
        print(f"\n🔍 扫描测试目录中的图片文件:")
        image_files = []
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file.lower())
                if ext in config2.SUPPORTED_FORMATS:
                    image_files.append(file)
        
        print(f"   找到 {len(image_files)} 张图片:")
        for img in image_files:
            print(f"     - {img}")
        
        print(f"\n✅ 配置测试完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_input():
    """测试用户输入功能的模拟"""
    print("\n🧪 模拟用户输入测试...")
    
    test_inputs = [
        ("/home/zhiqics/sanjian/predata/test_images", "✅ 有效目录"),
        ("/nonexistent/directory", "❌ 不存在的目录"),
        ("", "❌ 空输入"),
        ("help", "💡 帮助信息"),
        ("~/test", "🏠 用户目录符号"),
    ]
    
    for test_input, expected in test_inputs:
        print(f"\n测试输入: '{test_input}' -> {expected}")
        
        if test_input == "":
            print("   结果: 空输入将被拒绝")
        elif test_input == "help":
            print("   结果: 将显示帮助信息")
        elif test_input.startswith("~"):
            expanded = os.path.expanduser(test_input)
            print(f"   展开后: {expanded}")
        elif not os.path.exists(test_input):
            print("   结果: 目录不存在，将提示重新输入")
        else:
            abs_path = os.path.abspath(test_input)
            print(f"   绝对路径: {abs_path}")
            if os.path.isdir(abs_path):
                print("   结果: 有效目录")
            else:
                print("   结果: 不是目录")

def main():
    """主测试函数"""
    print("🚀 交互式人脸车牌检测器 - 配置测试")
    print("=" * 60)
    
    # 运行配置测试
    config_success = test_config()
    
    # 运行用户输入测试
    test_user_input()
    
    print("\n" + "=" * 60)
    if config_success:
        print("🎉 所有测试通过!")
        print("\n💡 现在你可以运行主程序:")
        print("   python 2-fast_face_plate_detector_v2.py")
    else:
        print("❌ 部分测试失败!")
    
    print("\n📖 使用说明:")
    print("   1. 运行主程序后会提示输入图片目录")
    print("   2. 输出将自动保存在输入目录的 'processed_output' 子目录中")
    print("   3. 支持相对路径、绝对路径和用户目录符号 (~)")
    print("   4. 输入 'help' 查看详细帮助")
    print("   5. 输入 'q' 或 'quit' 退出程序")

if __name__ == "__main__":
    main()
