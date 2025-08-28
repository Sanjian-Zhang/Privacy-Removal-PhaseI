#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门的4K WebM下载工具
优化WebM VP9/AV1编码下载，保持原始格式
"""

import os
import sys
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

def which_or_die(bin_name: str):
    """检查依赖是否存在"""
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    """执行命令的统一函数"""
    print("🚀", " ".join(str(x) for x in cmd))
    return subprocess.run(
        cmd,
        timeout=timeout,
        check=check,
        text=True,
        capture_output=capture
    )

def atomic_move(src: Path, dst: Path):
    """原子移动文件"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == 18:  # Invalid cross-device link
            shutil.copy2(src, dst)
            src.unlink()
        else:
            raise

def check_video_info(video_path: Path):
    """检查视频信息"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,bit_rate",
            "-show_entries", "format=format_name,size,duration",
            "-of", "csv=p=0",
            str(video_path)
        ]
        
        result = run(cmd, capture=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                # 视频流信息
                video_info = lines[0].split(',')
                width = int(video_info[0]) if video_info[0].isdigit() else 0
                height = int(video_info[1]) if video_info[1].isdigit() else 0
                fps_str = video_info[2] if len(video_info) > 2 else "0/1"
                codec = video_info[3] if len(video_info) > 3 else "unknown"
                bitrate = video_info[4] if len(video_info) > 4 else "unknown"
                
                # 格式信息
                format_info = lines[1].split(',')
                format_name = format_info[0] if format_info else "unknown"
                file_size = int(format_info[1]) if len(format_info) > 1 and format_info[1].isdigit() else 0
                duration = float(format_info[2]) if len(format_info) > 2 and format_info[2].replace('.', '').isdigit() else 0
                
                # 计算帧率
                try:
                    if "/" in fps_str:
                        num, den = map(int, fps_str.split("/"))
                        fps = num / den if den != 0 else 0
                    else:
                        fps = float(fps_str)
                except:
                    fps = 0
                
                return {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'codec': codec,
                    'bitrate': bitrate,
                    'format': format_name,
                    'size_mb': file_size / (1024*1024),
                    'duration': duration
                }
    except Exception as e:
        print(f"⚠️ 获取视频信息失败: {e}")
    
    return None

def download_4k_webm(url: str, out_path: Path, cookies: Path|None, with_audio: bool, 
                     timeout: int, retries: int):
    """下载4K WebM视频"""
    which_or_die("yt-dlp")
    
    # 创建临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="webm_"))
    tmp_file = tmp_dir / (out_path.name + ".part")

    # 4K WebM格式优先级列表
    webm_formats = [
        # YouTube特定格式代码（4K WebM）
        {
            "format": "337+251" if with_audio else "337",
            "desc": "4K 60fps VP9 WebM (337)" + ("+Opus音频(251)" if with_audio else "")
        },
        {
            "format": "401+251" if with_audio else "401", 
            "desc": "4K 60fps AV1 WebM (401)" + ("+Opus音频(251)" if with_audio else "")
        },
        {
            "format": "313+251" if with_audio else "313",
            "desc": "4K 30fps VP9 WebM (313)" + ("+Opus音频(251)" if with_audio else "")
        },
        # 通用选择器
        {
            "format": "bv*[height>=2160][vcodec*=vp9][ext=webm]+ba[ext=webm]/bv*[height>=2160][vcodec*=vp9][ext=webm]" if with_audio else "bv*[height>=2160][vcodec*=vp9][ext=webm]",
            "desc": "4K VP9 WebM 通用选择" + ("带音频" if with_audio else "")
        },
        {
            "format": "bv*[height>=2160][vcodec*=av01][ext=webm]+ba[ext=webm]/bv*[height>=2160][vcodec*=av01][ext=webm]" if with_audio else "bv*[height>=2160][vcodec*=av01][ext=webm]",
            "desc": "4K AV1 WebM 通用选择" + ("带音频" if with_audio else "")
        },
        {
            "format": "bv*[height>=2160][ext=webm]+ba[ext=webm]/bv*[height>=2160][ext=webm]" if with_audio else "bv*[height>=2160][ext=webm]",
            "desc": "任何4K WebM" + ("带音频" if with_audio else "")
        }
    ]

    # 构建基础命令
    base_cmd = [
        "yt-dlp",
        "-N", "8",                    # 并发分段
        "--no-part",                  # 不使用.part文件
        "--continue",                 # 断点续传
        "--restrict-filenames",       # 限制文件名字符
        "--merge-output-format", "webm",  # 强制输出WebM格式
        "--prefer-free-formats",      # 优先自由格式
        "--extractor-args", "youtube:player_client=web,ios,android",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", str(tmp_file),
    ]
    
    if cookies and cookies.exists():
        base_cmd += ["--cookies", str(cookies)]
        print("🍪 使用 cookies 文件")
    
    if not with_audio:
        base_cmd += ["--no-audio"]

    # 尝试各种格式
    for i in range(1, retries + 1):
        print(f"\n🎯 下载尝试 {i}/{retries}")
        
        for j, fmt_info in enumerate(webm_formats, 1):
            print(f"📺 尝试格式 {j}/{len(webm_formats)}: {fmt_info['desc']}")
            
            # 如果是重试且文件存在，先删除
            if tmp_file.exists():
                tmp_file.unlink()
            
            cmd = base_cmd + ["-f", fmt_info["format"], url]
            
            try:
                result = run(cmd, timeout=timeout)
                
                if result.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 1024:
                    # 下载成功，移动到目标位置
                    atomic_move(tmp_file, out_path)
                    
                    # 检查视频信息
                    print(f"✅ 下载成功：{out_path}")
                    video_info = check_video_info(out_path)
                    
                    if video_info:
                        print(f"📺 视频信息：")
                        print(f"   分辨率：{video_info['width']}x{video_info['height']}")
                        print(f"   帧率：{video_info['fps']:.1f} fps")
                        print(f"   编码：{video_info['codec']}")
                        print(f"   格式：{video_info['format']}")
                        print(f"   大小：{video_info['size_mb']:.1f} MB")
                        print(f"   时长：{video_info['duration']:.1f} 秒")
                        
                        # 验证是否为4K
                        is_4k = video_info['height'] >= 2160
                        is_webm = 'webm' in video_info['format'].lower()
                        
                        if is_4k and is_webm:
                            print("✅ 确认：4K WebM 视频下载成功")
                        else:
                            print(f"⚠️ 注意：4K({is_4k}) WebM({is_webm})")
                    
                    # 清理临时目录
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return True
                    
                else:
                    print(f"❌ 格式 {fmt_info['desc']} 下载失败")
                    
            except subprocess.TimeoutExpired:
                print("⏰ 下载超时")
            except Exception as e:
                print(f"❌ 下载异常：{e}")
    
    # 所有尝试都失败
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 所有WebM格式下载都失败")
    print("💡 建议：")
    print("   1. 检查视频是否有4K WebM格式")
    print("   2. 尝试使用浏览器cookies")
    print("   3. 检查网络连接")
    print("   4. 更新yt-dlp: pip install -U yt-dlp")
    return False

def main():
    parser = argparse.ArgumentParser(description="专门的4K WebM下载工具")
    parser.add_argument("url", help="YouTube视频链接")
    parser.add_argument("-o", "--output", default="./video_4k.webm", help="输出文件路径")
    parser.add_argument("-c", "--cookies", help="cookies文件路径")
    parser.add_argument("-a", "--audio", action="store_true", help="包含音频")
    parser.add_argument("-t", "--timeout", type=int, default=1800, help="下载超时时间（秒）")
    parser.add_argument("-r", "--retries", type=int, default=3, help="重试次数")
    
    args = parser.parse_args()
    
    print("===== 🎬 4K WebM 专用下载工具 =====")
    print("🌐 专门优化WebM VP9/AV1编码下载")
    print("📦 保持原始WebM格式和编码")
    print("🔧 支持断点续传和多重重试")
    print()
    
    url = args.url
    out_path = Path(args.output)
    cookies_path = Path(args.cookies) if args.cookies and Path(args.cookies).exists() else None
    
    print(f"📺 视频链接：{url}")
    print(f"💾 输出文件：{out_path}")
    print(f"🔊 包含音频：{'是' if args.audio else '否'}")
    print()
    
    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 开始下载
    success = download_4k_webm(
        url=url,
        out_path=out_path,
        cookies=cookies_path,
        with_audio=args.audio,
        timeout=args.timeout,
        retries=args.retries
    )
    
    if success:
        print(f"\n🎉 下载完成：{out_path}")
        sys.exit(0)
    else:
        print(f"\n❌ 下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
