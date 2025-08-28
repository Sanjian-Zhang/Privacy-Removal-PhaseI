#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import gc
import re
import time
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# ==================== 通用工具 ====================
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
                cmd.insert(1, "-loglevel"); cmd.insert(2, "error")
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

# ==================== ffmpeg 兼容兜底 ====================
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
    try:
        return json.loads(res.stdout or "{}")
    except Exception:
        return None

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
    info = ffprobe_json(path)
    if not info or "streams" not in info or not info["streams"]:
        print("⚠️ ffprobe 失败，直接转码到 H.264…")
        _transcode_to_h264(path); return
    fmt = (info.get("format", {}) or {}).get("format_name", "") or ""
    vcodec = (info["streams"][0] or {}).get("codec_name", "")
    is_mp4 = "mp4" in fmt
    is_h264 = (vcodec == "h264")
    if is_mp4 and is_h264:
        print("✅ 已兼容：MP4 + H.264"); return
    if (not is_mp4) and is_h264:
        fixed = path.with_suffix(".repack.mp4")
        res = run(["ffmpeg", "-i", str(path), "-c", "copy", "-movflags", "+faststart", "-y", str(fixed)])
        if res.returncode == 0 and fixed.exists() and fixed.stat().st_size > 0:
            atomic_move(fixed, path)
            print("🔁 已重封装为 MP4（无重压缩）")
            return ensure_ffmpeg_compatible(path)
    _transcode_to_h264(path)

# ==================== 下载（两阶段策略防止 .mp4.webm） ====================
def download_video(url: str, out_dir: Path, video_number: str) -> Path:
    which_or_die("yt-dlp"); which_or_die("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_"))
    stem = f"downloaded_video{video_number}"
    out_tmpl = str(tmp_dir / (stem + ".%(ext)s"))
    out_final = out_dir / f"{stem}.mp4"

    attempts = [
        ("bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]", None, "优先 MP4+M4A"),
        ("bv*+ba/b", "ffmpeg:-c:v copy -c:a aac -b:a 192k -movflags +faststart", "兜底：任意音频→AAC")
    ]
    for i, (fmt, pp, desc) in enumerate(attempts, 1):
        print(f"🎯 尝试 {i}/{len(attempts)}：{desc}")
        cmd = [
            "yt-dlp", "-N", "8",
            "--no-part", "--continue", "--restrict-filenames",
            "--merge-output-format", "mp4",
            "-o", out_tmpl, "-f", fmt, url
        ]
        if pp:
            cmd += ["--postprocessor-args", pp]
        res = run(cmd)
        if res.returncode == 0:
            cand = None
            for f in tmp_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".mov"):
                    if ".f" in f.name or f.name.endswith(".ytdl"):
                        continue
                    cand = f
            if cand:
                atomic_move(cand, out_final)
                print(f"✅ 下载完成：{out_final}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                ensure_ffmpeg_compatible(out_final)
                return out_final
        print("⚠️ 本次方案失败，切换下一种…")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 视频下载失败"); sys.exit(1)

# ==================== 抽帧 ====================
def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float, jpg_q: int, video_number: str) -> bool:
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"video{video_number}_frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vsync", "vfr",
        "-vf", f"fps={fps}",
        "-q:v", str(jpg_q),
        "-vcodec", "mjpeg",
        "-y", str(pattern)
    ]
    res = run(cmd)
    if res.returncode == 0:
        print(f"✅ 抽帧完成：{output_dir}")
        return True
    else:
        print("❌ 抽帧失败")
        return False

# ==================== 动态加载并调用你的分类器 ====================
def load_classifier_module(module_path: Path):
    """把 face_plate_classifier.py 当模块加载（不需要改你原文件）"""
    import importlib.util
    if not module_path.exists():
        print(f"❌ 找不到分类器文件：{module_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("face_plate_classifier", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_classifier(mod, frames_dir: Path, plate_model_path: Path):
    """
    关键点：在实例化 FacePlateClassifier 之前，直接重写它模块级全局常量，
    解决你原脚本里 INPUT_DIR/OUTPUT_BASE_DIR/模型路径写死的问题。
    """
    # 1) 改写输入/输出与模型路径
    mod.INPUT_DIR = str(frames_dir)
    mod.OUTPUT_BASE_DIR = str(frames_dir)
    mod.PLATE_MODEL_PATH = str(plate_model_path)

    # 2) 同步目录常量（它们在原模块里是基于 OUTPUT_BASE_DIR 定义的）
    mod.QUALIFIED_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "qualified")
    mod.INSUFFICIENT_SCORE_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "insufficient_score")
    mod.NO_CONTENT_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "no_content")
    mod.ANALYSIS_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "analysis")
    for d in [mod.QUALIFIED_DIR, mod.INSUFFICIENT_SCORE_DIR, mod.NO_CONTENT_DIR, mod.ANALYSIS_DIR]:
        os.makedirs(d, exist_ok=True)

    # 3) 运行
    clf = mod.FacePlateClassifier()
    clf.run()

# ==================== 交互式入口 ====================
def main():
    print("===== 🎬 交互式下载→抽帧→分类 工具 =====")
    url = input("请输入视频链接或本地文件路径: ").strip()
    if not url:
        print("❌ 必须输入链接或本地路径"); sys.exit(1)

    video_number = input("请输入视频编号（用于命名，如 38）: ").strip()
    if not video_number or not re.fullmatch(r"\d+", video_number):
        print("❌ 视频编号必须是数字"); sys.exit(1)

    fps_in = input("每秒抽帧数（默认=3，直接回车使用默认）: ").strip()
    fps = float(fps_in) if fps_in else 3.0

    jpg_in = input("JPG质量（默认=1，越小越清晰，2-31；直接回车=1）: ").strip()
    jpg_q = int(jpg_in) if jpg_in else 1

    # 你的目录布局（可按需修改）
    base_dir = Path("/home/zhiqics/sanjian/predata")
    video_dir = base_dir / "videos"
    frames_dir = base_dir / f"output_frames{video_number}"

    # 1) 获取视频
    downloaded = False
    if url.startswith("http"):
        video_path = download_video(url, video_dir, video_number)
        downloaded = True
    else:
        video_path = Path(url)
        if not video_path.exists():
            print(f"❌ 本地视频不存在：{video_path}"); sys.exit(1)
        ensure_ffmpeg_compatible(video_path)

    # 2) 抽帧
    ok = extract_frames_ffmpeg(
        video_path=video_path,
        output_dir=frames_dir,
        fps=fps,
        jpg_q=jpg_q,
        video_number=video_number
    )
    if not ok:
        sys.exit(1)

    # 3) 成功后删除“本次下载”的视频文件
    if downloaded:
        try:
            video_path.unlink(missing_ok=True)
            print(f"🧹 已删除下载视频：{video_path}")
            try:
                if video_dir.exists() and not any(video_dir.iterdir()):
                    video_dir.rmdir()
                    print(f"🧼 已清理空目录：{video_dir}")
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ 删除下载视频失败：{e}")

    # 4) 运行你的分类器（来自 face_plate_classifier.py）
    print("\n===== 🧠 开始图像分类（正脸/车牌/OCR） =====")
    # 分类器文件路径（默认：当前目录）
    default_classifier = Path.cwd() / "face_plate_classifier.py"
    c_in = input(f"分类器文件路径（回车默认：{default_classifier}）: ").strip()
    classifier_path = Path(c_in) if c_in else default_classifier

    # 车牌模型路径
    default_model = base_dir / "models/license_plate_detector.pt"
    m_in = input(f"车牌检测模型路径（回车默认：{default_model}）: ").strip()
    model_path = Path(m_in) if m_in else default_model
    if not model_path.exists():
        print(f"❌ 模型文件不存在：{model_path}"); sys.exit(1)

    # 动态加载并运行
    mod = load_classifier_module(classifier_path)
    run_classifier(mod, frames_dir, model_path)

if __name__ == "__main__":
    main()
