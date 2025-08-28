#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新抽帧工具 - 用于测试改进的抽帧功能
"""

import sys
from pathlib import Path

# 导入主抽帧函数
sys.path.append(str(Path(__file__).parent))
from down import extract_frames_ffmpeg, ffprobe_json

def main():
    print("===== 🔄 重新抽帧测试工具 =====")
    
    # 查找已有的视频文件
    video_dir = Path("/home/zhiqics/sanjian/predata/videos")
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            print("\n📹 找到的视频文件：")
            for i, vf in enumerate(video_files, 1):
                size_mb = vf.stat().st_size / (1024*1024)
                print(f"  {i}. {vf.name} ({size_mb:.1f} MB)")
            
            choice = input(f"\n选择视频文件 (1-{len(video_files)}): ").strip()
            try:
                video_path = video_files[int(choice)-1]
            except (ValueError, IndexError):
                print("❌ 选择无效")
                return
        else:
            print("❌ videos 目录中没有找到 MP4 文件")
            return
    else:
        print("❌ videos 目录不存在")
        return
    
    # 获取视频信息
    print(f"\n🔍 分析视频：{video_path.name}")
    info = ffprobe_json(video_path)
    if info and "streams" in info and info["streams"]:
        stream = info["streams"][0]
        width = stream.get("width", 0)
        height = stream.get("height", 0)
        print(f"📺 原视频分辨率：{width}x{height}")
    
    # 设置抽帧参数
    video_number = input("\n请输入新的视频编号（如 97）: ").strip()
    if not video_number:
        print("❌ 必须输入视频编号")
        return
    
    fps = float(input("每秒抽帧数（默认=1）: ").strip() or "1")
    jpg_q = int(input("JPG质量（默认=1，最高质量）: ").strip() or "1")
    
    print("\n📐 分辨率选项：")
    print("  1. 保持原分辨率")
    print("  2. 1920x1080")
    print("  3. 1280x720")
    print("  4. 自定义")
    
    res_choice = input("选择 (默认=1): ").strip() or "1"
    if res_choice == "2":
        max_resolution = "1920x1080"
    elif res_choice == "3":
        max_resolution = "1280x720"
    elif res_choice == "4":
        custom = input("输入分辨率 (如 2560x1440): ").strip()
        max_resolution = custom if "x" in custom else "1920x1080"
    else:
        max_resolution = "4096x4096"  # 基本不限制，保持原分辨率
    
    # 开始抽帧
    output_dir = Path(f"/home/zhiqics/sanjian/predata/output_frames{video_number}")
    
    try:
        extract_frames_ffmpeg(
            video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            jpg_q=jpg_q,
            video_number=video_number,
            max_resolution=max_resolution
        )
        
        # 检查结果
        frames = list(output_dir.glob("*.jpg"))
        if frames:
            print(f"\n🎉 成功生成 {len(frames)} 张图片")
            # 显示第一张图片的信息
            import subprocess
            result = subprocess.run(["file", str(frames[0])], 
                                  capture_output=True, text=True)
            print(f"📸 第一张图片：{result.stdout.strip()}")
        else:
            print("❌ 没有生成图片")
            
    except Exception as e:
        print(f"❌ 抽帧失败：{e}")

if __name__ == "__main__":
    main()
