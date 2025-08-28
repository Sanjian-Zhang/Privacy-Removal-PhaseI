#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import tempfile
import re
import time
from pathlib import Path
from typing import Optional

# ============ 工具函数 ============
def which_or_die(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    """统一执行命令；对 ffmpeg/ffprobe 降噪但保留 stats。"""
    if isinstance(cmd, (list, tuple)) and cmd:
        head = str(cmd[0])
        if head.endswith("ffmpeg") or head.endswith("ffprobe") or head in ("ffmpeg", "ffprobe"):
            if "-hide_banner" not in cmd:
                cmd.insert(1, "-hide_banner")
            if "-loglevel" not in cmd:
                cmd.insert(1, "-loglevel")
                cmd.insert(2, "error")
            if (head.endswith("ffmpeg") or head == "ffmpeg") and "-stats" not in cmd:
                cmd.insert(3, "-stats")
    print("🚀", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, timeout=timeout, check=check, text=True, capture_output=capture)

def atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == 18:  # cross-device
            shutil.copy2(src, dst); src.unlink()
        else:
            raise

# ============ 兼容性兜底（确保可抽帧） ============
def ffprobe_json(path: Path):
    which_or_die("ffprobe")
    res = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=format_name:stream=codec_name,avg_frame_rate,nb_frames",
        "-of", "json", str(path)
    ], capture=True)
    if res.returncode != 0:
        return None
    import json
    try:
        return json.loads(res.stdout or "{}")
    except Exception:
        return None

def check_video_hdr_status(video_path: Path):
    """检查视频是否为HDR格式"""
    try:
        # 使用ffprobe检查颜色空间和传输特性
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=color_space,color_transfer,color_primaries,pix_fmt",
            "-of", "csv=p=0",
            str(video_path)
        ]
        
        result = run(cmd, capture=True)
        if result.returncode != 0:
            return False, "无法检测"
        
        output = result.stdout.strip()
        if not output:
            return False, "无颜色信息"
        
        # 分析颜色信息
        parts = output.split(',')
        if len(parts) >= 4:
            color_space = parts[0].lower() if parts[0] else ""
            color_transfer = parts[1].lower() if parts[1] else ""
            color_primaries = parts[2].lower() if parts[2] else ""
            pix_fmt = parts[3].lower() if parts[3] else ""
        else:
            return False, "颜色信息不完整"
        
        # HDR特征检测
        hdr_indicators = [
            # 颜色传输特性
            "smpte2084",  # PQ (Perceptual Quantization)
            "arib-std-b67",  # HLG (Hybrid Log-Gamma)
            "smpte428",  # DCI-P3
            "bt2020",
            # 像素格式
            "yuv420p10",  # 10-bit
            "yuv422p10",
            "yuv444p10",
            "p010",
            # 颜色空间
            "bt2020nc",
            "bt2020c"
        ]
        
        is_hdr = any(indicator in color_transfer.lower() for indicator in hdr_indicators) or \
                 any(indicator in color_space.lower() for indicator in hdr_indicators) or \
                 any(indicator in color_primaries.lower() for indicator in hdr_indicators) or \
                 any(indicator in pix_fmt.lower() for indicator in hdr_indicators)
        
        info = f"空间:{color_space}, 传输:{color_transfer}, 原色:{color_primaries}, 格式:{pix_fmt}"
        
        return is_hdr, info
        
    except Exception as e:
        return False, f"检测错误: {e}"

def verify_non_hdr_video(video_path: Path):
    """验证视频不是HDR格式"""
    print("🔍 验证非HDR状态...")
    is_hdr, hdr_info = check_video_hdr_status(video_path)
    
    print(f"📊 颜色信息: {hdr_info}")
    
    if is_hdr:
        print("⚠️ 警告: 检测到HDR特征!")
        print("💡 建议:")
        print("   1. 重新下载非HDR版本")
        print("   2. 或使用ffmpeg转换为SDR")
        
        choice = input("是否继续使用此HDR视频? (y/N): ").strip().lower()
        if choice != 'y':
            print("❌ 用户选择不使用HDR视频")
            return False
    else:
        print("✅ 确认: 非HDR视频，适合处理")
    
    return True

def _transcode_to_h264(path: Path):
    which_or_die("ffmpeg")
    out = path.with_suffix(".h264.mp4")
    res = run([
        "ffmpeg", "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-y", str(out)
    ])
    if res.returncode != 0 or (not out.exists()) or out.stat().st_size == 0:
        print("❌ 转码失败"); sys.exit(1)
    atomic_move(out, path)
    print("🧩 已转码为 H.264 MP4")

def ensure_ffmpeg_compatible(path: Path):
    """确保容器=mp4 且视频编码=h264；否则重封装或转码。"""
    info = ffprobe_json(path)
    if not info or "streams" not in info or not info["streams"]:
        print("⚠️ ffprobe 失败，直接转码到 H.264…")
        _transcode_to_h264(path); return
    fmt = (info.get("format", {}) or {}).get("format_name", "") or ""
    vcodec = (info["streams"][0] or {}).get("codec_name", "")
    is_mp4 = "mp4" in fmt
    is_h264 = (vcodec == "h264")
    if is_mp4 and is_h264:
        print("✅ 已兼容：MP4 + H.264")
        return
    if (not is_mp4) and is_h264:
        # 仅重封装
        fixed = path.with_suffix(".repack.mp4")
        res = run(["ffmpeg", "-i", str(path), "-c", "copy", "-movflags", "+faststart", "-y", str(fixed)])
        if res.returncode == 0 and fixed.exists() and fixed.stat().st_size > 0:
            atomic_move(fixed, path)
            print("🔁 已重封装为 MP4（无重压缩）")
            return ensure_ffmpeg_compatible(path)
    # 其他情况 → 转码
    _transcode_to_h264(path)

def check_4k_60fps_available(url):
    """检查视频是否有4K 60fps格式"""
    try:
        print("🔍 预检查：检测4K 60fps格式可用性...")
        cmd = [
            "yt-dlp",
            "--list-formats", 
            "--extractor-args", "youtube:player_client=web,ios,android",
            url
        ]
        
        result = run(cmd, capture=True)
        if result.returncode != 0:
            return False, "无法获取格式列表"
        
        output = result.stdout or ""
        
        # 检查是否有4K 60fps格式
        import re
        has_4k_60fps = bool(re.search(r'21\d{2}p.*60', output) or 
                           re.search(r'4K.*60', output) or
                           re.search(r'3840x2160.*60', output))
        
        if has_4k_60fps:
            return True, "✅ 检测到4K 60fps格式"
        else:
            return False, "❌ 未检测到4K 60fps格式"
            
    except Exception as e:
        return False, f"预检查失败: {e}"

# ============ 下载视频（两阶段策略防止 .mp4.webm） ============
def download_video(url: str, out_dir: Path, video_number: str) -> Path:
    which_or_die("yt-dlp"); which_or_die("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预检查4K 60fps格式
    if url.startswith("http"):
        is_available, message = check_4k_60fps_available(url)
        print(message)
        if not is_available:
            print("💡 建议：")
            print("   1. 检查视频是否真的有4K 60fps格式")
            print("   2. 使用 check_youtube_formats.py 查看详细格式")
            print("   3. 尝试其他4K 60fps视频")
            choice = input("是否仍要尝试下载? (y/N): ").strip().lower()
            if choice != 'y':
                sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_"))
    stem = f"downloaded_video{video_number}"
    out_tmpl = str(tmp_dir / (stem + ".%(ext)s"))
    
    # 根据下载结果确定最终文件名（支持WebM格式）
    out_final_base = out_dir / stem
    # 不预设扩展名，根据实际下载结果确定

    # 检查 yt-dlp 版本并更新
    print("🔧 检查 yt-dlp 版本...")
    version_res = run(["yt-dlp", "--version"], capture=True)
    if version_res.returncode == 0:
        print(f"📦 当前版本：{version_res.stdout.strip()}")
    
    # 尝试更新 yt-dlp
    print("🔄 尝试更新 yt-dlp...")
    update_res = run(["yt-dlp", "-U"], capture=True)
    if update_res.returncode == 0 and "Updated" in update_res.stdout:
        print("✅ yt-dlp 已更新")
    else:
        print("ℹ️ yt-dlp 已是最新版本或更新失败")

    attempts = [
        # 1) 4K WebM VP9 格式 - 优先WebM容器，排除HDR
        ("337+251", None, "4K 60fps VP9 WebM (337+251) [非HDR]"),
        # 2) 4K WebM AV1 格式 - 新一代编码，排除HDR
        ("401+251", None, "4K 60fps AV1 WebM (401+251) [非HDR]"),
        # 3) 通用4K WebM格式选择器 - 明确排除HDR
        ("bv*[height>=2160][fps>=60][ext=webm][vcodec!*=hdr][vcodec!*=hev][vcodec!*=dv]+ba[ext=webm]/bv*[height>=2160][fps>=60][ext=webm][vcodec!*=hdr]+ba", None, "4K 60fps WebM 通用 [排除HDR]"),
        # 4) 4K VP9 WebM (备选编号) - 非HDR
        ("313+251", None, "4K 30fps VP9 WebM (313+251) [非HDR]"),
        # 5) 任何4K WebM格式 - 排除HDR/DV
        ("bv*[height>=2160][ext=webm][vcodec!=hdr][vcodec!=hevc_hdr][vcodec!=dv]+ba/bv*[height>=2160][ext=webm][vcodec!=hdr]", None, "任何4K WebM [排除HDR]"),
        # 6) YouTube 4K AVC 格式 (备选) - H.264不支持HDR
        ("315+140", None, "4K 60fps AVC MP4 (315+140) [H.264非HDR]"),
        # 7) 通用4K 60帧选择器 - 排除HDR格式
        ("bv*[height>=2160][fps>=60][vcodec!*=hdr][vcodec!*=dv][ext=webm]+ba/bv*[height>=2160][fps>=60][vcodec!*=hdr]+ba/b[height>=2160][fps>=60][vcodec!*=hdr]", None, "4K 60帧 WebM优先 [排除HDR]"),
        # 8) 最佳4K 60帧 - 明确排除HDR/DV/HLG
        ("bestvideo[height>=2160][fps>=60][vcodec!=hdr][vcodec!=hevc_hdr][vcodec!=dv][vcodec!=hlg]+bestaudio/best[height>=2160][fps>=60][vcodec!=hdr]", None, "最佳4K 60帧 [严格排除HDR]")
    ]

    # 检查是否有 cookies 文件
    cookies_file = Path("/home/zhiqics/sanjian/predata/cookies.txt")
    use_cookies = cookies_file.exists() and cookies_file.stat().st_size > 0

    for i, (fmt, pp, desc) in enumerate(attempts, 1):
        print(f"🎯 尝试 {i}/{len(attempts)}：{desc}")
        cmd = [
            "yt-dlp",
            "--no-part", "--continue",
            "--restrict-filenames",
            # 不强制合并为MP4，保持原格式（特别是WebM）
            "--keep-video",  # 保持原视频格式
            "--retry-sleep", "5",
            "--fragment-retries", "10",
            "--extractor-retries", "3",
            "--socket-timeout", "30",
            "--extractor-args", "youtube:player_client=web,ios,android",
            "--extractor-args", "youtube:skip=dash,hls",
            "-o", out_tmpl,
            "-f", fmt
        ]
        
        # 添加 cookies 支持
        if use_cookies:
            cmd.extend(["--cookies", str(cookies_file)])
            print("🍪 使用 cookies 文件")
        
        # 添加用户代理
        cmd.extend([
            "--user-agent", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ])
        
        cmd.append(url)
        
        if pp:
            cmd += ["--postprocessor-args", pp]
        
        res = run(cmd)
        if res.returncode == 0:
            # 找到合并后的成品
            cand = None
            for f in tmp_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm"):
                    if ".f" in f.name or f.name.endswith(".ytdl"):
                        continue
                    cand = f
                    break
            if cand and cand.stat().st_size > 1024:  # 至少 1KB
                # 根据实际下载的文件格式确定最终文件名
                actual_ext = cand.suffix.lower()
                out_final = out_final_base.with_suffix(actual_ext)
                
                atomic_move(cand, out_final)
                print(f"✅ 下载完成：{out_final} ({actual_ext[1:].upper()}格式)")
                
                # 验证是否真的是4K 60fps，且非HDR
                print("🔍 验证视频格式...")
                info = ffprobe_json(out_final)
                if info and "streams" in info and info["streams"]:
                    stream = info["streams"][0]
                    width = stream.get("width", 0)
                    height = stream.get("height", 0)
                    codec = stream.get("codec_name", "unknown")
                    
                    # 检查帧率
                    fps_str = stream.get("r_frame_rate", "0/1")
                    try:
                        if "/" in fps_str:
                            num, den = map(int, fps_str.split("/"))
                            fps = num / den if den != 0 else 0
                        else:
                            fps = float(fps_str)
                    except:
                        fps = 0
                    
                    is_4k = height >= 2160
                    is_60fps = fps >= 59
                    
                    print(f"📺 实际格式：{width}x{height}, {fps:.1f}fps, 编码:{codec}")
                    print(f"📦 容器格式：{actual_ext[1:].upper()}")
                    
                    if is_4k and is_60fps:
                        print("✅ 确认：4K 60fps视频")
                        
                        # 🚀 HDR检测
                        if not verify_non_hdr_video(out_final):
                            print("❌ HDR视频被拒绝，尝试下一种格式...")
                            out_final.unlink()  # 删除HDR视频
                            continue
                            
                    else:
                        print(f"⚠️ 警告：不是标准4K 60fps (4K: {is_4k}, 60fps: {is_60fps})")
                        
                        # 即使不是完美的4K 60fps，也检查是否HDR
                        is_hdr, hdr_info = check_video_hdr_status(out_final)
                        if is_hdr:
                            print("❌ 检测到HDR格式，跳过此视频")
                            out_final.unlink()
                            continue
                
                shutil.rmtree(tmp_dir, ignore_errors=True)
                
                # 只对非WebM格式进行兼容性修复
                if actual_ext != ".webm":
                    ensure_ffmpeg_compatible(out_final)
                else:
                    print("🔧 保持WebM原始格式，跳过MP4转换")
                
                return out_final
        print("⚠️ 本次方案失败，切换下一种…")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 无法下载4K 60fps视频")
    print("💡 可能的原因：")
    print("   1. 该视频没有4K 60fps格式")
    print("   2. 视频有地区限制或需要登录")
    print("   3. 网络连接问题")
    print("   4. 需要使用浏览器 cookies（导出到 cookies.txt）")
    print("📝 建议：")
    print("   - 检查视频是否真的有4K 60fps格式")
    print("   - 尝试其他4K 60fps视频链接")
    print("   - 手动更新 yt-dlp: pip install -U yt-dlp")
    sys.exit(1)

# ============ 性能监控 ============
def get_video_duration(video_path: Path) -> float:
    """获取视频时长"""
    try:
        result = run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path)
        ], capture=True)
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
    except:
        pass
    return 0.0

def monitor_frame_extraction(output_dir: Path, video_number: str, estimated_frames: int, duration: float):
    """监控抽帧进度"""
    import threading
    import glob
    
    def progress_monitor():
        last_count = 0
        start_time = time.time()
        
        while True:
            try:
                # 统计当前生成的帧数
                pattern = str(output_dir / f"video{video_number}_frame_*.jpg")
                current_count = len(glob.glob(pattern))
                
                if current_count > last_count:
                    elapsed = time.time() - start_time
                    fps_speed = current_count / elapsed if elapsed > 0 else 0
                    progress = (current_count / estimated_frames * 100) if estimated_frames > 0 else 0
                    
                    print(f"\r🎬 进度: {current_count}/{estimated_frames} ({progress:.1f}%) | 速度: {fps_speed:.1f} 帧/秒", end="", flush=True)
                    last_count = current_count
                
                time.sleep(2)  # 每2秒更新一次
                
            except:
                break
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

# ============ 抽帧 ============
def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float, jpg_q: int, video_number: str, max_resolution: str = "4096x4096"):
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 🚀 预先获取视频信息用于性能监控
    print("📊 分析视频信息...")
    duration = get_video_duration(video_path)
    estimated_frames = int(duration * fps) if duration > 0 else 0
    
    if duration > 0:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"⏱️  视频时长: {minutes}:{seconds:02d} | 预计抽取: {estimated_frames} 帧")
    
    # 🔧 修复：改进视频分辨率获取
    info = ffprobe_json(video_path)
    width, height = 1920, 1080  # 默认值
    
    if info and "streams" in info and info["streams"]:
        stream = info["streams"][0]
        detected_width = stream.get("width", 0)
        detected_height = stream.get("height", 0)
        
        # 如果检测到有效分辨率，使用检测值
        if detected_width > 0 and detected_height > 0:
            width, height = detected_width, detected_height
            print(f"📺 检测到分辨率：{width}x{height}")
        else:
            # 尝试用另一种方法获取分辨率
            print("🔧 使用备用方法检测分辨率...")
            probe_res = run([
                "ffprobe", "-v", "error", 
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(video_path)
            ], capture=True)
            
            if probe_res.returncode == 0 and probe_res.stdout:
                try:
                    w_h = probe_res.stdout.strip().split(',')
                    if len(w_h) >= 2 and w_h[0].isdigit() and w_h[1].isdigit():
                        width, height = int(w_h[0]), int(w_h[1])
                        print(f"📺 备用检测分辨率：{width}x{height}")
                except:
                    print(f"⚠️ 无法检测分辨率，使用默认：{width}x{height}")
            else:
                print(f"⚠️ 无法检测分辨率，使用默认：{width}x{height}")
    else:
        print(f"⚠️ 无法获取视频流信息，使用默认分辨率：{width}x{height}")
    
    # 🚀 高质量高速构建视频滤镜
    max_w, max_h = map(int, max_resolution.split('x'))
    if width > max_w or height > max_h:
        # 使用高质量缩放算法，保持画质
        scale_filter = f"scale='if(gt(iw,ih),min({max_w},iw),-2)':'if(gt(iw,ih),-2,min({max_h},ih))':flags=lanczos"
        vf = f"fps={fps},{scale_filter}"
        print(f"🔽 分辨率过大，将缩放到最大 {max_resolution} (高质量算法)")
    else:
        # 保持原分辨率
        vf = f"fps={fps}"
        print(f"✅ 保持原分辨率 {width}x{height}")
    
    # 固定命名：video{编号}_frame_000001.jpg
    pattern = output_dir / f"video{video_number}_frame_%06d.jpg"
    
    # � 修复：优化FFmpeg参数组合
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        # 🚀 多线程解码优化
        "-threads", "0",              # 使用所有CPU核心
        "-thread_type", "slice",      # 启用切片多线程
        # � 修复：使用兼容的帧提取方式
        "-vf", vf,                    # 视频滤镜（包含fps）
        # 🚀 保持最高画质设置
        "-q:v", str(jpg_q),           # 保持用户指定的质量
        "-pix_fmt", "yuvj420p",       # 确保兼容性
        "-vcodec", "mjpeg",
        # 🚀 I/O优化
        "-f", "image2",               # 明确指定输出格式
        "-y", str(pattern)
    ]
    
    print(f"🎬 开始高速抽帧：{fps} FPS，最高质量={jpg_q}")
    print("🚀 已启用多线程优化，保持原画质")
    
    # 🚀 启动性能监控
    start_time = time.time()
    if estimated_frames > 0:
        monitor_thread = monitor_frame_extraction(output_dir, video_number, estimated_frames, duration)
    
    res = run(cmd)
    
    # 🚀 计算最终性能统计
    end_time = time.time()
    elapsed = end_time - start_time
    
    if res.returncode == 0:
        # 统计实际生成的帧数
        import glob
        pattern_glob = str(output_dir / f"video{video_number}_frame_*.jpg")
        actual_frames = len(glob.glob(pattern_glob))
        
        print(f"\n✅ 抽帧完成：{output_dir}")
        print(f"📊 总耗时：{elapsed:.1f}秒")
        print(f"📊 实际帧数：{actual_frames}张")
        
        if elapsed > 0:
            fps_speed = actual_frames / elapsed
            print(f"📊 处理速度：{fps_speed:.1f} 帧/秒")
            
            if duration > 0:
                real_time_ratio = duration / elapsed
                print(f"📊 实时比率：{real_time_ratio:.1f}x (高于1.0表示比实时播放快)")
        
        # 检查生成的第一张图片信息
        first_frame = output_dir / f"video{video_number}_frame_000001.jpg"
        if first_frame.exists():
            frame_info_res = run(["file", str(first_frame)], capture=True)
            if frame_info_res.returncode == 0:
                print(f"📸 生成的图片信息：{frame_info_res.stdout.strip()}")
    else:
        print("\n❌ 抽帧失败")
        print("🔧 尝试使用兼容模式...")
        
        # 🚀 备用兼容模式
        fallback_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", str(jpg_q),
            "-y", str(pattern)
        ]
        
        print("🔧 使用简化参数重试...")
        fallback_res = run(fallback_cmd)
        
        if fallback_res.returncode == 0:
            import glob
            pattern_glob = str(output_dir / f"video{video_number}_frame_*.jpg")
            actual_frames = len(glob.glob(pattern_glob))
            print(f"✅ 兼容模式成功！生成 {actual_frames} 帧")
        else:
            print("❌ 兼容模式也失败"); sys.exit(1)

# ============ 交互式入口 ============
def main():
    print("===== 🎬 4K 60fps 视频抽帧工具 (WebM优化版 + 非HDR) =====")
    print("⚠️  注意：本工具专门下载4K 60fps视频，优先WebM格式！")
    print("📺 支持格式：WebM (VP9/AV1)、MP4 (H.264)")
    print("🔧 WebM格式将保持原始编码，提供更好的压缩效率")
    print("� 自动排除HDR格式，确保兼容性")
    print("�📦 如果视频没有4K 60fps格式将下载失败")
    print("🚀 抽帧性能优化：多线程处理 + 实时进度监控")
    print()
    
    url = input("请输入视频链接或本地文件路径: ").strip()
    if not url:
        print("❌ 必须输入链接或本地路径"); sys.exit(1)

    video_number = input("请输入视频编号（用于命名，如 88）: ").strip()
    if not video_number or not re.fullmatch(r"\d+", video_number):
        print("❌ 视频编号必须是数字"); sys.exit(1)

    fps_in = input("每秒抽帧数（默认=3，直接回车使用默认）: ").strip()
    fps = float(fps_in) if fps_in else 3.0

    jpg_in = input("JPG质量（默认=1最高质量，2-31；直接回车=1）: ").strip()
    jpg_q = int(jpg_in) if jpg_in else 1
    
    if jpg_q == 1:
        print("✨ 使用最高画质 (质量=1)，抽帧可能较慢但画质最佳")
    else:
        print(f"⚡ 使用质量={jpg_q}，在画质和速度间平衡")

    # 新增：分辨率控制 - 针对4K优化
    print("\n📐 4K分辨率设置：")
    print("  1. 保持原4K分辨率（推荐）")
    print("  2. 强制标准4K (3840x2160)")
    print("  3. 自定义分辨率")
    res_choice = input("选择分辨率选项（默认=1）: ").strip()
    
    if res_choice == "2":
        max_resolution = "3840x2160"
    elif res_choice == "3":
        custom_res = input("输入自定义分辨率（格式：宽x高，如 4096x2160）: ").strip()
        if re.match(r"\d+x\d+", custom_res):
            max_resolution = custom_res
        else:
            print("❌ 分辨率格式错误，使用默认保持原分辨率")
            max_resolution = "4096x4096"
    else:
        max_resolution = "4096x4096"  # 基本不限制，保持原分辨率

    # 根目录可按需修改
    base_dir = Path("/home/zhiqics/sanjian/predata")
    video_dir = base_dir / "videos"
    frames_dir = base_dir / f"output_frames{video_number}"

    # 1) 获取视频文件
    if url.startswith("http"):
        video_path = download_video(url, video_dir, video_number)
    else:
        video_path = Path(url)
        if not video_path.exists():
            print(f"❌ 本地视频不存在：{video_path}"); sys.exit(1)
        
        # 本地视频也检查HDR状态
        print("🔍 检查本地视频HDR状态...")
        if not verify_non_hdr_video(video_path):
            print("❌ 本地视频为HDR格式，建议使用非HDR视频")
            sys.exit(1)
        
        # 本地视频也做一次兼容性兜底
        ensure_ffmpeg_compatible(video_path)

    # 2) 抽帧
    print(f"\n🚀 开始抽帧处理...")
    print(f"💾 输出目录: {frames_dir}")
    extract_frames_ffmpeg(
        video_path=video_path,
        output_dir=frames_dir,
        fps=fps,
        jpg_q=jpg_q,
        video_number=video_number,
        max_resolution=max_resolution
    )

if __name__ == "__main__":
    main()
