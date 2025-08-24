#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom training script - for training our processed test images
"""

import os
import sys
import shutil
from source.train import ciagan_exp

def setup_training_data():
    """
    Prepare data structure for training
    """
    # Create data directory structure required for training
    train_data_dir = "/home/zhiqics/sanjian/baseline/ciagan/train_data/"
    source_dir = "/home/zhiqics/sanjian/baseline/ciagan/processed_output/"
    
    # Create directories
    os.makedirs(train_data_dir, exist_ok=True)
    
    # Copy processed data
    for subdir in ['clr', 'lndm', 'msk', 'orig']:
        src_path = os.path.join(source_dir, subdir)
        dst_path = os.path.join(train_data_dir, subdir)
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")
    
    return train_data_dir

def train_small_dataset():
    """
    使用我们的小数据集进行训练测试
    """
    print("=== CIAGAN 训练开始 ===")
    
    # 准备训练数据
    train_data_dir = setup_training_data()
    print(f"训练数据目录: {train_data_dir}")
    
    # 检查数据
    identity_count = len(os.listdir(os.path.join(train_data_dir, 'clr')))
    print(f"发现 {identity_count} 个身份")
    
    for identity in os.listdir(os.path.join(train_data_dir, 'clr')):
        image_count = len([f for f in os.listdir(os.path.join(train_data_dir, 'clr', identity)) 
                          if f.endswith('.jpg')])
        print(f"身份 {identity}: {image_count} 张图片")
    
    # 配置训练参数
    config_updates = {
        'TRAIN_PARAMS': {
            'ARCH_NUM': 'unet_flex',        # 生成器架构
            'ARCH_SIAM': 'resnet_siam',     # Siamese网络架构
            'EPOCH_START': 0,               # 开始轮次
            'EPOCHS_NUM': 20,               # 总训练轮次 (小数据集用较少轮次)
            'LEARNING_RATE': 0.0001,        # 学习率
            'FILTER_NUM': 16,               # 过滤器数量 (较小以适应小数据集)
            
            'ITER_CRITIC': 1,               # Discriminator训练次数
            'ITER_GENERATOR': 2,            # Generator训练次数
            'ITER_SIAMESE': 1,              # Siamese网络训练次数
            
            'GAN_TYPE': 'lsgan',            # GAN损失类型
            'FLAG_GPU': True,               # 使用GPU
        },
        'DATA_PARAMS': {
            'DATA_PATH': train_data_dir,    # 数据路径
            'DATA_SET': '',                 # 数据集名称 (空字符串表示直接使用DATA_PATH)
            'LABEL_NUM': identity_count,    # 身份数量
            'BATCH_SIZE': 1,                # 批次大小 (小数据集用1)
            'WORKERS_NUM': 2,               # 数据加载器工作进程数
            'IMG_SIZE': 64,                 # 图像尺寸 (较小以加快训练)
            'FLAG_DATA_AUGM': False,        # 数据增强 (小数据集不增强)
        },
        'OUTPUT_PARAMS': {
            'RESULT_PATH': '/home/zhiqics/sanjian/baseline/ciagan/results/',
            'MODEL_PATH': '/home/zhiqics/sanjian/baseline/ciagan/models/',
            'LOG_ITER': 2,                  # 每2步输出一次日志
            'SAVE_EPOCH': 5,                # 每5轮保存一次模型
            'SAVE_CHECKPOINT': 10,          # 每10轮保存检查点
            'SAVE_IMAGES': True,            # 保存生成的图像
            'PROJECT_NAME': 'ciagan_test',
            'EXP_TRY': 'small_dataset',
            'COMMENT': "Testing with processed test images",
        }
    }
    
    print("=== 训练配置 ===")
    print(f"训练轮次: {config_updates['TRAIN_PARAMS']['EPOCHS_NUM']}")
    print(f"批次大小: {config_updates['DATA_PARAMS']['BATCH_SIZE']}")
    print(f"图像尺寸: {config_updates['DATA_PARAMS']['IMG_SIZE']}")
    print(f"身份数量: {config_updates['DATA_PARAMS']['LABEL_NUM']}")
    print(f"学习率: {config_updates['TRAIN_PARAMS']['LEARNING_RATE']}")
    
    # 创建输出目录
    os.makedirs(config_updates['OUTPUT_PARAMS']['RESULT_PATH'], exist_ok=True)
    os.makedirs(config_updates['OUTPUT_PARAMS']['MODEL_PATH'], exist_ok=True)
    
    # 开始训练
    print("\n=== 开始训练 ===")
    try:
        result = ciagan_exp.run(config_updates=config_updates)
        print("=== 训练完成 ===")
        print(f"结果保存在: {config_updates['OUTPUT_PARAMS']['RESULT_PATH']}")
        print(f"模型保存在: {config_updates['OUTPUT_PARAMS']['MODEL_PATH']}")
        return result
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    train_small_dataset()
