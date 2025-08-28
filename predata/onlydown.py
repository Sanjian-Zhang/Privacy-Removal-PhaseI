import os
import sys
import shutil
import argparse
import subprocess
import tempfile
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

def ffmpeg_convert(input_path: Path, output_path: Path, target_format: str = "mp4", 
                  target_codec: str = "libx264", crf: int = 23):
    """使用ffmpeg转换视频格式"""
    which_or_die("ffmpeg")
    
    print(f"🔄 开始ffmpeg转换：{input_path.name} -> {output_path.name}")
    print(f"📺 目标格式：{target_format}, 编码：{target_codec}, CRF：{crf}")
    
    cmd = [
        "ffmpeg", "-y",  # 覆盖输出文件
        "-i", str(input_path),
        "-c:v", target_codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]
    
    try:
        result = run(cmd, timeout=3600)  # 1小时超时
        if result.returncode == 0:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ ffmpeg转换完成：{output_path}（{file_size_mb:.1f} MB）")
            return True
        else:
            print(f"❌ ffmpeg转换失败，返回码：{result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ ffmpeg转换超时")
        return False
    except Exception as e:
        print(f"❌ ffmpeg转换异常：{e}")
        return False

# ------------ 核心流程 ------------
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
        "-o", str(tmp_file),
        url,
    ]
    if cookies:
        base_cmd += ["--cookies", str(cookies)]
    
    # 专门设置为4K WebM视频格式 - 优先VP9/AV1编码
    if with_audio:
        # 优先下载4K WebM视频+音频，VP9/AV1编码优先
        format_str = (
            f"bv*[height={fmt_height}][vcodec*=av01][ext=webm]+ba[ext=webm]/"     # AV1 4K WebM + WebM音频
            f"bv*[height={fmt_height}][vcodec*=vp9][ext=webm]+ba[ext=webm]/"      # VP9 4K WebM + WebM音频
            f"bv*[height={fmt_height}][ext=webm]+ba[ext=webm]/"                   # 任何4K WebM + WebM音频
            f"bv*[height={fmt_height}][vcodec*=av01]+ba/"                         # AV1 4K + 音频
            f"bv*[height={fmt_height}][vcodec*=vp9]+ba/"                          # VP9 4K + 音频  
            f"bv*[height={fmt_height}][vcodec*=avc1]+ba/"                         # H264 4K + 音频
            f"bv*[height={fmt_height}]+ba/"                                       # 任何4K + 音频
            f"bv*[height>={fmt_height}]+ba/"                                      # 4K以上 + 音频
            f"best[height>={fmt_height}]"                                         # 最后选择
        )
        base_cmd += ["-f", format_str]
        # 根据格式字符串判断优先输出格式
        if "webm" in format_str:
            base_cmd += ["--merge-output-format", "webm"]
        else:
            base_cmd += ["--merge-output-format", "mp4"]
    else:
        # 只下载4K WebM视频，优先VP9/AV1编码
        format_str = (
            f"bv*[height={fmt_height}][vcodec*=av01][ext=webm]/"                  # AV1 WebM
            f"bv*[height={fmt_height}][vcodec*=vp9][ext=webm]/"                   # VP9 WebM
            f"bv*[height={fmt_height}][ext=webm]/"                                # 任何4K WebM
            f"bv*[height={fmt_height}][vcodec*=avc1][ext=mp4]/"                   # H264 MP4
            f"bv*[height={fmt_height}]/"                                          # 任何4K格式
            f"best[height>={fmt_height}]"                                         # 最后选择
        )
        base_cmd += ["-f", format_str]
        base_cmd += ["--no-audio"]

    last_err = None
    for i in range(1, retries+1):
        print(f"🎯 正在下载4K视频（第 {i}/{retries} 次重试）：{url}")
        print(f"📺 目标分辨率：{fmt_height}p (4K)")
        # 如果是重试且临时文件已存在，先删除避免断点续传问题
        if i > 1 and tmp_file.exists():
            tmp_file.unlink()
        try:
            res = run(base_cmd, timeout=timeout)
            if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 0:
                atomic_move(tmp_file, out_path)
                # 获取视频信息
                file_size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"✅ 4K视频下载完成：{out_path}（{file_size_mb:.1f} MB）")
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
    print("❌ 4K视频下载失败，已用尽重试次数")
    print("💡 提示：可能该视频没有4K版本，请检查视频源或降低分辨率要求")
    sys.exit(1)

# ------------ CLI 入口 ------------
def main():
    parser = argparse.ArgumentParser(description="YouTube 4K WebM下载工具")
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=28ZjrtD_iL0",)
    parser.add_argument("--out_video", default="./downloaded_video89.webm")  # 默认WebM格式
    parser.add_argument("--cookies", default="cookies.txt")
    parser.add_argument("--with_audio", action="store_true", help="下载时合并音频（默认不带音频）")
    parser.add_argument("--height", type=int, default=2160, help="最高分辨率高度限制（4K=2160）")
    parser.add_argument("--dl_timeout", type=int, default=1200, help="下载超时时间（秒）")
    parser.add_argument("--dl_retries", type=int, default=2, help="下载重试次数")
    # ffmpeg转换选项
    parser.add_argument("--convert", action="store_true", help="下载后使用ffmpeg转换格式")
    parser.add_argument("--target_codec", default="libx264", help="目标视频编码（默认：libx264）")
    parser.add_argument("--crf", type=int, default=23, help="CRF质量参数（默认：23）")
    parser.add_argument("--converted_suffix", default="_converted", help="转换后文件名后缀")
    args = parser.parse_args()

    url = args.url
    out_video = Path(args.out_video)
    cookies_path = Path(args.cookies) if args.cookies and Path(args.cookies).exists() else None

    # 下载视频（带断点续传与重试）
    download_youtube(
        url=url,
        out_path=out_video,
        cookies=cookies_path,
        with_audio=args.with_audio,
        fmt_height=args.height,
        timeout=args.dl_timeout,
        retries=args.dl_retries
    )
    
    # 如果需要，使用ffmpeg转换格式
    if args.convert:
        print("\n" + "="*50)
        print("🎬 开始ffmpeg后处理")
        
        # 生成转换后的文件名
        stem = out_video.stem
        suffix = out_video.suffix
        converted_name = f"{stem}{args.converted_suffix}{suffix}"
        converted_path = out_video.parent / converted_name
        
        # 执行转换
        if ffmpeg_convert(out_video, converted_path, 
                         target_codec=args.target_codec, 
                         crf=args.crf):
            print(f"🎯 原始文件：{out_video}")
            print(f"✨ 转换文件：{converted_path}")
            
            # 询问是否保留原文件
            original_size = out_video.stat().st_size / (1024 * 1024)
            converted_size = converted_path.stat().st_size / (1024 * 1024)
            print(f"📊 大小对比：原始 {original_size:.1f}MB -> 转换后 {converted_size:.1f}MB")
        else:
            print("❌ ffmpeg转换失败，保留原始下载文件")

if __name__ == "__main__":
    main()
