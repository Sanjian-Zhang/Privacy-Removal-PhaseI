import os
import sys
import shutil
import argparse
import subprocess
import tempfile
import re
from pathlib import Path

# ------------ 工具函数 ------------
def which_or_die(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    # 默认把 ffmpeg 的输出降噪；yt-dlp 保留进度
    env = os.environ.copy()
    if "ffmpeg" in cmd[0]:
        if "-hide_banner" not in cmd:
            cmd.insert(1, "-hide_banner")
        if "-loglevel" not in cmd:
            cmd.insert(1, "-loglevel")
            cmd.insert(2, "error")
        if "-stats" not in cmd:
            cmd.insert(3, "-stats")
    print("🚀", " ".join(str(x) for x in cmd))
    return subprocess.run(
        cmd,
        timeout=timeout,
        check=check,
        text=True,
        capture_output=capture
    )

def atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)  # 原子移动
    except OSError as e:
        if e.errno == 18:  # Invalid cross-device link
            # 跨设备移动，使用复制+删除
            import shutil
            shutil.copy2(src, dst)
            src.unlink()
        else:
            raise

# ------------ YouTube下载功能 ------------
def download_youtube(url: str, out_path: Path, cookies: Path|None, with_audio: bool, fmt_height: int, timeout: int, retries: int):
    which_or_die("yt-dlp")
    # 临时文件，成功后原子替换
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_"))
    tmp_file = tmp_dir / (out_path.name + ".part.mp4")

    base_cmd = [
        "yt-dlp",
        "-N", "8",                 # 并发分段
        "--no-part",               # 直接输出到目标（我们已用 tmp）
        "--continue",              # 断点续传
        "--no-warnings",
        "--restrict-filenames",
        "--no-write-subs",         # 不写入字幕文件
        "--no-write-auto-subs",    # 不写入自动生成的字幕
        "--no-embed-subs",         # 不嵌入字幕到视频
        "--no-write-info-json",    # 不写入信息文件
        "--no-write-description",  # 不写入描述文件
        "--no-write-thumbnail",    # 不写入缩略图
        "-o", str(tmp_file),
        url,
    ]
    if cookies:
        base_cmd += ["--cookies", str(cookies)]
    # 选择格式：带音频则合并；不带音频只下视频轨
    if with_audio:
        base_cmd += ["-f", f"bv*[height<={fmt_height}][fps<=60]+ba/b[height<={fmt_height}]"]
        base_cmd += ["--merge-output-format", "mp4"]
    else:
        base_cmd += ["-f", f"bv*[height<={fmt_height}][fps<=60][ext=mp4]"]
        base_cmd += ["--no-audio"]

    last_err = None
    for i in range(1, retries+1):
        print(f"🎯 正在下载（第 {i}/{retries} 次重试）：{url}")
        # 如果是重试且临时文件已存在，先删除避免断点续传问题
        if i > 1 and tmp_file.exists():
            tmp_file.unlink()
        try:
            res = run(base_cmd, timeout=timeout)
            if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 0:
                atomic_move(tmp_file, out_path)
                print(f"✅ 下载完成：{out_path}（{out_path.stat().st_size/(1024*1024):.1f} MB）")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return
            else:
                last_err = f"返回码 {res.returncode}"
        except subprocess.TimeoutExpired:
            last_err = "下载超时"
        except Exception as e:
            last_err = f"异常：{e}"
        print(f"⚠️ 本次失败：{last_err}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 下载失败，已用尽重试次数")
    sys.exit(1)

# ------------ 抽帧功能 ------------
def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float, start: float|None, duration: float|None,
                          scale_width: int|None, img_format: str, png_level: int, jpg_q: int, threads: int):
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 从视频路径中提取数字部分作为videoX命名
    video_name = video_path.stem
    # 使用正则表达式提取文件名中的数字
    match = re.search(r'(\d+)', video_name)
    if match:
        video_number = match.group(1)
        video_prefix = f"video{video_number}"
    else:
        # 如果没有找到数字，使用完整文件名
        video_prefix = video_name
    pattern = output_dir / f"{video_prefix}_frame_%06d.{img_format.lower()}"

    vf_chain = []
    # 抽帧（vfr 在可变帧率上更稳）
    vf_chain.append(f"fps={fps}")
    # 可选时间段
    ss_args, t_args = [], []
    if start is not None:
        ss_args = ["-ss", str(start)]
    if duration is not None:
        t_args = ["-t", str(duration)]
        
    # 可选尺寸（只指定宽度，保持比例）
    if scale_width:
        vf_chain.append(f"scale={scale_width}:-2:flags=bicubic")

    cmd = ["ffmpeg"] + ss_args + [
        "-i", str(video_path),
        "-vsync", "vfr",
        "-threads", str(threads),
        "-map_metadata", "-1",  # 去掉元数据
        "-an"                   # 不处理音频
    ] + t_args + [
        "-vf", ",".join(vf_chain),
        "-y"
    ]

    # 输出格式参数
    if img_format.lower() == "png":
        # 0-9，越大越小但更慢；对 4K 建议 3~6 平衡
        cmd += ["-compression_level", str(png_level)]
    elif img_format.lower() in ("jpg", "jpeg"):
        # 2-31，越小越清晰；用 -q:v 代替 -quality
        cmd += ["-q:v", str(jpg_q)]
        # 强制使用 mjpeg 编码器更兼容
        cmd += ["-vcodec", "mjpeg"]
    else:
        print("❌ 仅支持 png / jpg")
        sys.exit(1)

    cmd += [str(pattern)]
    res = run(cmd)
    if res.returncode == 0:
        print(f"✅ 抽帧完成：{output_dir}")
    else:
        print("❌ 抽帧失败")
        sys.exit(1)

# ------------ CLI 入口 ------------
def main():
    parser = argparse.ArgumentParser(description="YouTube视频下载+抽帧工具")
    
    # 下载相关参数
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=MAj6y23vNuU", help="YouTube视频URL")
    parser.add_argument("--out_video", default="./video47.mp4", help="输出视频文件路径")
    parser.add_argument("--cookies", default="cookies.txt", help="cookies文件路径")
    parser.add_argument("--with_audio", action="store_true", help="下载时合并音频（默认不带音频）")
    parser.add_argument("--height", type=int, default=2160, help="最高分辨率高度限制")
    parser.add_argument("--dl_timeout", type=int, default=1200, help="下载超时时间（秒）")
    parser.add_argument("--dl_retries", type=int, default=2, help="下载重试次数")
    
    # 抽帧相关参数
    parser.add_argument("--out_frames", default=None, help="输出帧目录（默认根据视频名生成）")
    parser.add_argument("--fps", type=float, default=3.0, help="每秒抽取帧数")
    parser.add_argument("--start", type=float, default=None, help="抽帧起始秒（可选）")
    parser.add_argument("--duration", type=float, default=None, help="抽帧时长秒（可选）")
    parser.add_argument("--scale_width", type=int, default=None, help="可选：统一缩放到指定宽度（如 3840 表示 4K）")
    parser.add_argument("--format", choices=["png", "jpg"], default="jpg", help="帧输出格式")
    parser.add_argument("--png_level", type=int, default=4, help="PNG 压缩级别 0-9，越大越慢越小")
    parser.add_argument("--jpg_q", type=int, default=1, help="JPG 质量系数 2-31，越小越清晰")
    parser.add_argument("--threads", type=int, default=0, help="ffmpeg 线程数（0=自动）")
    
    # 流程控制
    parser.add_argument("--skip_download", action="store_true", help="跳过下载，直接抽帧")
    parser.add_argument("--skip_extract", action="store_true", help="跳过抽帧，只下载")
    
    args = parser.parse_args()

    # 处理路径
    out_video = Path(args.out_video)
    cookies_path = Path(args.cookies) if args.cookies and Path(args.cookies).exists() else None
    
    # 如果没有指定抽帧输出目录，根据视频文件名生成
    if args.out_frames is None:
        video_stem = out_video.stem
        match = re.search(r'(\d+)', video_stem)
        if match:
            video_number = match.group(1)
            out_frames = Path(f"./output_frames{video_number}")
        else:
            out_frames = Path(f"./output_frames_{video_stem}")
    else:
        out_frames = Path(args.out_frames)

    # 步骤1：下载视频
    if not args.skip_download:
        print("=" * 50)
        print("📥 开始下载视频...")
        print("=" * 50)
        download_youtube(
            url=args.url,
            out_path=out_video,
            cookies=cookies_path,
            with_audio=args.with_audio,
            fmt_height=args.height,
            timeout=args.dl_timeout,
            retries=args.dl_retries
        )
    else:
        if not out_video.exists():
            print(f"❌ 视频文件不存在：{out_video}")
            sys.exit(1)
        print(f"⏭️  跳过下载，使用现有视频：{out_video}")

    # 步骤2：抽帧
    if not args.skip_extract:
        print("=" * 50)
        print("🎬 开始抽帧...")
        print("=" * 50)
        extract_frames_ffmpeg(
            video_path=out_video,
            output_dir=out_frames,
            fps=args.fps,
            start=args.start,
            duration=args.duration,
            scale_width=args.scale_width,
            img_format=args.format,
            png_level=args.png_level,
            jpg_q=args.jpg_q,
            threads=args.threads
        )
    else:
        print("⏭️  跳过抽帧")

    print("=" * 50)
    print("🎉 全部完成！")
    print(f"📹 视频文件：{out_video}")
    if not args.skip_extract:
        print(f"🖼️  帧文件夹：{out_frames}")
    print("=" * 50)

if __name__ == "__main__":
    main()
