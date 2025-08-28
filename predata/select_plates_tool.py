#!/usr/bin/env python3
"""
选择指定目录的图片，挑选出有2个清晰车牌的图片
使用命令行参数指定输入和输出目录
"""

import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from select_clear_plates import process_images

def main():
    parser = argparse.ArgumentParser(description="挑选有2个清晰车牌的图片")
    parser.add_argument("input_dir", help="输入图片目录路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default=None)
    parser.add_argument("--single-plate", action="store_true", 
                       help="允许单个车牌（默认要求2个车牌）")
    parser.add_argument("--move", action="store_true", 
                       help="移动文件而不是复制（默认复制）")
    parser.add_argument("--conf", type=float, default=0.4, 
                       help="检测置信度阈值 (默认: 0.4)")
    parser.add_argument("--sharpness", type=float, default=100, 
                       help="清晰度阈值 (默认: 100)")
    
    args = parser.parse_args()
    
    # 设置输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return 1
    
    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        plates_type = "single_plate" if args.single_plate else "two_plates"
        output_dir = input_dir.parent / f"{input_dir.name}_{plates_type}_selected"
    
    # 确定车牌要求
    require_two_plates = not args.single_plate
    plates_desc = "至少1个" if args.single_plate else "至少2个"
    
    print("🚀 车牌选择工具")
    print("="*50)
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 车牌要求: {plates_desc}清晰车牌")
    print(f"📋 文件操作: {'移动' if args.move else '复制'}")
    print(f"🔍 检测置信度: {args.conf}")
    print(f"✨ 清晰度阈值: {args.sharpness}")
    print("="*50)
    
    try:
        # 运行处理
        process_images(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            model_path=None,  # 使用默认模型
            conf_threshold=args.conf,
            sharpness_threshold=args.sharpness,
            contrast_threshold=20,
            copy_original=not args.move,  # copy_original与move相反
            batch_size=30,
            edge_density_threshold=0.05,
            text_clarity_threshold=15,
            require_two_plates=require_two_plates
        )
        
        print("✅ 处理完成！")
        return 0
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
