#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuDNN兼容性修复工具
解决 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH 错误
"""

import os
import sys
import logging
from pathlib import Path

def fix_cudnn_compatibility():
    """修复cuDNN兼容性问题"""
    print("🔧 正在修复cuDNN兼容性...")
    
    # 设置环境变量
    env_fixes = {
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'TF_CPP_MIN_LOG_LEVEL': '2',  # 减少TensorFlow日志
        'CUDA_LAUNCH_BLOCKING': '1',  # 同步CUDA操作便于调试
    }
    
    for key, value in env_fixes.items():
        os.environ[key] = value
        print(f"  ✅ 设置 {key}={value}")
    
    try:
        import torch
        if torch.cuda.is_available():
            # 清理GPU缓存
            torch.cuda.empty_cache()
            device_count = torch.cuda.device_count()
            print(f"  🚀 检测到 {device_count} 个GPU设备")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {name} ({memory:.1f}GB)")
            
            # 设置内存分配策略
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%显存
                print("  ⚙️ 设置GPU内存使用限制为80%")
        else:
            print("  ⚠️ CUDA不可用")
            
    except ImportError:
        print("  ⚠️ PyTorch未安装")
    
    # 尝试修复常见的cuDNN问题
    try:
        import torch.backends.cudnn as cudnn
        if torch.cuda.is_available():
            # 禁用benchmark模式可能解决版本不匹配问题
            cudnn.benchmark = False
            cudnn.deterministic = True
            print("  🔒 设置cuDNN为确定性模式")
    except Exception as e:
        print(f"  ⚠️ cuDNN设置警告: {e}")

def create_conda_environment_script():
    """创建conda环境设置脚本"""
    script_content = """#!/bin/bash
# cuDNN兼容性修复脚本

echo "🔧 修复cuDNN兼容性问题..."

# 设置环境变量
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TF_CPP_MIN_LOG_LEVEL=2

echo "✅ 环境变量已设置"

# 检查CUDA状态
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ nvidia-smi 未找到"
fi

echo "🎯 现在可以运行Python脚本了"
"""
    
    script_path = Path("/home/zhiqics/sanjian/predata/fix_cudnn.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # 添加执行权限
    import stat
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"✅ 创建cuDNN修复脚本: {script_path}")
    print("使用方法: source fix_cudnn.sh && python your_script.py")

def check_video_file(video_path: str):
    """检查视频文件状态"""
    path = Path(video_path)
    
    print(f"\n🔍 检查视频文件: {path}")
    
    if not path.exists():
        print("❌ 文件不存在")
        return False
    
    # 文件大小
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"📁 文件大小: {size_mb:.1f} MB")
    
    # 文件头检查
    try:
        with open(path, 'rb') as f:
            header = f.read(100)
        
        # 检查是否是有效的MP4文件
        if b'ftyp' in header[:20]:
            print("✅ 文件头正常 (MP4格式)")
        else:
            print("⚠️ 文件头异常")
            print(f"前20字节: {header[:20].hex()}")
    
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 使用ffprobe检查
    try:
        import subprocess
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            if 'format' in info:
                duration = info['format'].get('duration', 'unknown')
                format_name = info['format'].get('format_name', 'unknown')
                print(f"✅ ffprobe检查通过: 格式={format_name}, 时长={duration}s")
                return True
        
        print(f"❌ ffprobe失败: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ ffprobe检查异常: {e}")
        return False

def main():
    print("🚀 cuDNN兼容性修复工具")
    print("=" * 50)
    
    # 修复cuDNN
    fix_cudnn_compatibility()
    
    # 创建修复脚本
    create_conda_environment_script()
    
    # 检查问题视频
    problem_videos = [
        "/home/zhiqics/sanjian/predata/downloaded_video36.mp4",
        "/home/zhiqics/sanjian/predata/downloaded_video38.mp4"
    ]
    
    for video in problem_videos:
        check_video_file(video)
    
    print("\n" + "=" * 50)
    print("🎯 建议解决方案:")
    print("1. video36 和 video38 可能下载损坏，建议重新下载")
    print("2. 使用 video40 测试抽帧功能")
    print("3. 运行前先执行: source fix_cudnn.sh")
    print("4. 如果GPU问题持续，可以添加 --no-gpu 参数使用CPU")

if __name__ == "__main__":
    main()
