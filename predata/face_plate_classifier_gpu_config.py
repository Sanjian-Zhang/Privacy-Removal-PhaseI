#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速配置文件
提供了多种GPU加速选项和配置
"""

import torch

# GPU加速配置选项
class GPUConfig:
    """GPU配置类"""
    
    # 基本设置
    USE_GPU = True                      # 是否使用GPU
    AUTO_SELECT_GPU = True              # 自动选择最优GPU
    GPU_DEVICE_ID = 0                   # 手动指定GPU ID（当AUTO_SELECT_GPU=False时）
    
    # 性能优化
    ENABLE_TORCH_OPTIMIZATION = True    # 启用PyTorch优化
    MIXED_PRECISION = False             # 混合精度训练（实验性）
    
    # 内存管理
    GPU_MEMORY_FRACTION = 0.8          # GPU内存使用比例
    ENABLE_MEMORY_GROWTH = True        # 启用内存增长
    
    # 批处理设置
    BATCH_PROCESSING = False           # 批量处理（实验性）
    BATCH_SIZE = 4                     # 批量大小
    
    @staticmethod
    def get_optimal_device():
        """获取最优GPU设备"""
        if not GPUConfig.USE_GPU or not torch.cuda.is_available():
            return 'cpu'
        
        if not GPUConfig.AUTO_SELECT_GPU:
            if GPUConfig.GPU_DEVICE_ID < torch.cuda.device_count():
                return f'cuda:{GPUConfig.GPU_DEVICE_ID}'
            else:
                return 'cpu'
        
        # 自动选择显存最多且利用率最低的GPU
        best_device = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i
            except Exception:
                continue
        
        return f'cuda:{best_device}'
    
    @staticmethod
    def setup_gpu_optimizations():
        """设置GPU优化"""
        if GPUConfig.ENABLE_TORCH_OPTIMIZATION:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def get_device_info(device: str) -> dict:
        """获取设备信息"""
        if 'cuda' in device:
            device_id = int(device.split(':')[1]) if ':' in device else 0
            props = torch.cuda.get_device_properties(device_id)
            return {
                'name': props.name,
                'total_memory': props.total_memory / 1024**3,
                'allocated_memory': torch.cuda.memory_allocated(device_id) / 1024**3,
                'cached_memory': torch.cuda.memory_reserved(device_id) / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}"
            }
        else:
            return {'name': 'CPU', 'type': 'cpu'}

# 使用示例
if __name__ == "__main__":
    # 获取最优设备
    device = GPUConfig.get_optimal_device()
    print(f"Selected device: {device}")
    
    # 获取设备信息
    info = GPUConfig.get_device_info(device)
    print(f"Device info: {info}")
    
    # 设置优化
    GPUConfig.setup_gpu_optimizations()
