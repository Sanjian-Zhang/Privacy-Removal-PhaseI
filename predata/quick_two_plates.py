#!/usr/bin/env python3
"""
直接运行车牌选择 - 挑选有2个清晰车牌的图片
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入我们的车牌选择模块
from select_clear_plates import process_images

def main():
    # 设置路径
    input_dir = "/home/zhiqics/sanjian/predata/test_images"  # 测试图片目录
    output_dir = "/home/zhiqics/sanjian/predata/two_plates_result"  # 输出目录
    
    print("🚀 开始挑选有2个清晰车牌的图片...")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    try:
        # 运行车牌选择处理
        process_images(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=None,  # 使用默认YOLOv8模型
            conf_threshold=0.4,  # 检测置信度
            sharpness_threshold=100,  # 清晰度阈值
            contrast_threshold=20,    # 对比度阈值
            copy_original=True,  # 复制而不是移动文件
            batch_size=20,  # 批处理大小
            edge_density_threshold=0.05,  # 边缘密度阈值
            text_clarity_threshold=15,  # 文字清晰度阈值
            require_two_plates=True  # 要求至少2个清晰车牌
        )
        
        print("✅ 处理完成！")
        print(f"📁 请查看输出目录: {output_dir}")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
