#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIAGAN 推理脚本 - 使用预训练模型进行身份匿名化
"""

import os
import sys
import argparse

def run_anonymization(data_path, model_path, output_path, num_identities=1):
    """
    运行身份匿名化
    """
    print("=== CIAGAN 身份匿名化 ===")
    print(f"数据路径: {data_path}")
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"身份数量: {num_identities}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ 数据目录不存在: {data_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 运行推理
    try:
        # 添加源码路径到Python路径
        import sys
        import os
        source_path = os.path.join(os.path.dirname(__file__), 'source')
        sys.path.insert(0, source_path)
        
        from test import run_inference
        
        # 去掉 .pth 扩展名
        model_name = model_path.replace('.pth', '')
        
        run_inference(
            data_path=data_path,
            num_folders=num_identities,
            model_path=model_name,
            output_path=output_path
        )
        
        print("✅ 身份匿名化完成!")
        print(f"结果保存在: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIAGAN 身份匿名化')
    parser.add_argument('--data', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/processed_output/',
                       help='处理过的数据路径')
    parser.add_argument('--model', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/pretrained_models/modelG.pth',
                       help='预训练模型路径')
    parser.add_argument('--output', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/anonymized_output/',
                       help='输出路径')
    parser.add_argument('--ids', type=int, default=1,
                       help='身份数量')
    
    args = parser.parse_args()
    
    run_anonymization(args.data, args.model, args.output, args.ids)
