#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - GPU加速版正脸和车牌检测分类器
简化版本，最小化输出
"""

import os
import sys

# 设置环境变量启用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main():
    """快速启动主函数"""
    try:
        from face_plate_classifier_improved import FacePlateClassifier, Config
        
        # 创建配置
        config = Config()
        
        print(f"🚀 启动GPU加速分类器...")
        print(f"📁 输入: {config.INPUT_DIR}")
        print(f"💻 设备: GPU {config.GPU_DEVICE_ID}")
        
        # 运行分类器
        classifier = FacePlateClassifier(config)
        classifier.run()
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
