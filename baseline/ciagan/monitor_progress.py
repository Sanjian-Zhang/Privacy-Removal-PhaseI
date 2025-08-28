#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import glob

def monitor_progress():
    output_dir = "/home/zhiqics/sanjian/dataset/images/Processed/ciagan"
    
    print("=== CIAGAN Processing Progress Monitor ===")
    print("Press Ctrl+C to exit monitoring")
    print()
    
    try:
        while True:
            if os.path.exists(output_dir):
                processed_files = glob.glob(os.path.join(output_dir, "anon_*.jpg"))
                count = len(processed_files)
                
                print(f"\rProcessed: {count} images", end="", flush=True)
                
                if count > 0:
                    recent_files = sorted(processed_files)[-3:]
                    print(f"  (Latest: {', '.join([os.path.basename(f) for f in recent_files])})")
                else:
                    print("  (Waiting for processing to start...)")
            else:
                print(f"\rWaiting for output directory creation...", end="", flush=True)
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
        if os.path.exists(output_dir):
            final_count = len(glob.glob(os.path.join(output_dir, "anon_*.jpg")))
            print(f"Final processed count: {final_count} images")

if __name__ == "__main__":
    monitor_progress()
