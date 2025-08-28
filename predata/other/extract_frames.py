import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# ------------ 工具函数 ------------
def which_or_die(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    # 默认把 ffmpeg 的输出降噪
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

# ------------ 抽帧功能 ------------
def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float, start: float|None, duration: float|None,
                          scale_width: int|None, img_format: str, png_level: int, jpg_q: int, threads: int):
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 从视频路径中提取数字部分作为videoX命名
    import re
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
    parser = argparse.ArgumentParser(description="FFmpeg 视频抽帧工具")
    parser.add_argument("--video_path", default="/home/zhiqics/sanjian/predata/downloaded_video36.mp4", help="输入视频文件路径")
    parser.add_argument("--out_frames", default="/home/zhiqics/sanjian/predata/output_frames36", help="输出帧目录")
    parser.add_argument("--fps", type=float, default=3.0, help="每秒抽取帧数")
    parser.add_argument("--start", type=float, default=None, help="抽帧起始秒（可选）")
    parser.add_argument("--duration", type=float, default=None, help="抽帧时长秒（可选）")
    parser.add_argument("--scale_width", type=int, default=None, help="可选：统一缩放到指定宽度（如 3840 表示 4K）")
    parser.add_argument("--format", choices=["png", "jpg"], default="jpg", help="帧输出格式")
    parser.add_argument("--png_level", type=int, default=4, help="PNG 压缩级别 0-9，越大越慢越小")
    parser.add_argument("--jpg_q", type=int, default=1, help="JPG 质量系数 2-31，越小越清晰")
    parser.add_argument("--threads", type=int, default=0, help="ffmpeg 线程数（0=自动）")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists(): 
        print(f"❌ 视频文件不存在：{video_path}")
        sys.exit(1)

    out_frames = Path(args.out_frames)

    # 执行抽帧
    extract_frames_ffmpeg(
        video_path=video_path,
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

if __name__ == "__main__":
    main()
