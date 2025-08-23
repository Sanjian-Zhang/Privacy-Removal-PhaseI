#!/usr/bin/env python3
"""
GaRNet完整使用示例
这个脚本展示了如何使用GaRNet进行文本去除
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# 添加CODE目录到路径
sys.path.append('./CODE')
from model import GaRNet

def setup_garnet():
    """初始化GaRNet模型"""
    print("Setting up GaRNet...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型 (注意：预训练模型是用in_channels=3训练的)
    model = GaRNet(in_channels=3)
    model_path = "./WEIGHTS/GaRNet/saved_model.pth"
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print("✗ Model file not found. Please run download_model.py first")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, device

def run_example():
    """运行示例"""
    print("\n" + "="*50)
    print("GaRNet Scene Text Removal Example")
    print("="*50)
    
    # 检查示例文件
    example_img = "./DATA/EXAMPLE/IMG/img_302.jpg"
    example_txt = "./DATA/EXAMPLE/TXT/img_302.txt"
    output_path = "./example_output.jpg"
    
    if not os.path.exists(example_img):
        print("✗ Example image not found")
        return False
    
    if not os.path.exists(example_txt):
        print("✗ Example text file not found")
        return False
    
    print(f"✓ Input image: {example_img}")
    print(f"✓ Text coordinates: {example_txt}")
    print(f"✓ Output will be saved to: {output_path}")
    
    # 设置模型
    model, device = setup_garnet()
    if model is None:
        return False
    
    # 运行推理
    print("\nRunning inference...")
    try:
        # 这里可以添加具体的推理代码
        # 目前使用现有的inference脚本
        import subprocess
        result = subprocess.run([
            "python", "./CODE/inference.py", 
            "--gpu" if device == "cuda" else "",
            "--result_path", "./example_results",
            "--image_path", "./DATA/EXAMPLE/IMG",
            "--box_path", "./DATA/EXAMPLE/TXT"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Inference completed successfully")
            if os.path.exists("./example_results/img_302.jpg"):
                print(f"✓ Output image saved to: ./example_results/img_302.jpg")
                return True
            else:
                print("✗ Output image not found")
                return False
        else:
            print(f"✗ Inference failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

def print_usage_info():
    """打印使用信息"""
    print("\n" + "="*50)
    print("GaRNet Usage Information")
    print("="*50)
    print("\n1. Model Architecture:")
    print("   - GaRNet是一个场景文本去除模型")
    print("   - 使用Gated Attention机制")
    print("   - 支持GPU加速")
    
    print("\n2. 文件结构:")
    print("   ├── CODE/              # 模型和推理代码")
    print("   ├── DATA/EXAMPLE/      # 示例数据")
    print("   ├── WEIGHTS/           # 预训练模型")
    print("   └── garnet_env/        # Python虚拟环境")
    
    print("\n3. 主要功能:")
    print("   - inference.py: 图像推理")
    print("   - eval.py: 模型评估")
    print("   - model.py: 模型定义")
    
    print("\n4. 输入格式:")
    print("   - 图像: JPG/PNG格式")
    print("   - 文本框: TXT文件，每行包含4个点的坐标")
    
    print("\n5. 运行命令示例:")
    print("   cd CODE")
    print("   python inference.py --gpu --image_path ../DATA/EXAMPLE/IMG")

if __name__ == "__main__":
    print_usage_info()
    
    # 运行示例
    success = run_example()
    
    if success:
        print(f"\n✓ GaRNet示例运行成功!")
        print("你现在可以:")
        print("1. 查看输出结果图像")
        print("2. 使用自己的图像进行测试")
        print("3. 调整参数优化结果")
    else:
        print(f"\n✗ 示例运行失败，请检查错误信息")
        
    print(f"\n虚拟环境位置: {os.path.abspath('./garnet_env')}")
    print("激活虚拟环境: source garnet_env/bin/activate")
