#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控 CIAGAN 批量处理进度
"""

import os
import time
import glob

def monitor_progress():
    """
    监控处理进度
    """
    output_dir = "/home/zhiqics/sanjian/dataset/images/Processed/ciagan"
    
    print("=== CIAGAN 处理进度监控 ===")
    print("按 Ctrl+C 退出监控")
    print()
    
    try:
        while True:
            if os.path.exists(output_dir):
                # 统计已处理的文件
                processed_files = glob.glob(os.path.join(output_dir, "anon_*.jpg"))
                count = len(processed_files)
                
                print(f"\r已处理: {count} 张图片", end="", flush=True)
                
                # 如果有文件，显示最新的几个
                if count > 0:
                    recent_files = sorted(processed_files)[-3:]
                    print(f"  (最新: {', '.join([os.path.basename(f) for f in recent_files])})")
                else:
                    print("  (等待处理开始...)")
            else:
                print("\r等待输出目录创建...", end="", flush=True)
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        if os.path.exists(output_dir):
            final_count = len(glob.glob(os.path.join(output_dir, "anon_*.jpg")))
            print(f"最终处理数量: {final_count} 张图片")

if __name__ == "__main__":
    monitor_progress()
