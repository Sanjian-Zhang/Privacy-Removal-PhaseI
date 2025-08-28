#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化版本的图片处理器
专门针对内存受限环境优化
"""

import psutil
import gc
import time
import os

def check_memory_and_wait():
    """检查内存使用情况并等待"""
    while True:
        # 获取当前内存使用情况
        memory = psutil.virtual_memory()
        
        print(f"📊 系统内存状态:")
        print(f"   总内存: {memory.total / 1024**3:.1f} GB")
        print(f"   已使用: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)")
        print(f"   可用: {memory.available / 1024**3:.1f} GB")
        
        # 检查Python进程
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'python' in proc.info['name'].lower() and '2-fast_face_plate_detector' in ' '.join(proc.cmdline()):
                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                    cpu_percent = proc.info['cpu_percent']
                    print(f"   🐍 Python进程 {proc.info['pid']}: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # 检查是否有输出文件生成
        output_dirs = [
            "/home/zhiqics/sanjian/predata/output_frames70/processed_output",
            "/home/zhiqics/sanjian/predata/output_frames70/processed_output/high_score_images",
            "/home/zhiqics/sanjian/predata/output_frames70/processed_output/unique_high_score_images"
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                file_count = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                if file_count > 0:
                    print(f"   📁 发现输出: {output_dir} ({file_count} 个文件)")
        
        print(f"⏰ {time.strftime('%H:%M:%S')} - 继续监控中...")
        print("-" * 60)
        
        time.sleep(30)  # 每30秒检查一次

if __name__ == "__main__":
    print("🔍 开始监控图片处理进度...")
    print("=" * 60)
    
    try:
        check_memory_and_wait()
    except KeyboardInterrupt:
        print("\n⏹️ 监控已停止")
