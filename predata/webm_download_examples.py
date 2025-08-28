#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4K WebM下载使用示例
"""

import subprocess
import sys
from pathlib import Path

def example_usage():
    """示例用法"""
    print("===== 🎬 4K WebM下载工具使用示例 =====")
    print()
    
    script_path = Path(__file__).parent / "download_4k_webm.py"
    
    examples = [
        {
            "desc": "下载4K WebM视频（仅视频流）",
            "cmd": f"python3 {script_path} 'https://www.youtube.com/watch?v=EXAMPLE' -o ./test_4k.webm"
        },
        {
            "desc": "下载4K WebM视频（包含音频）", 
            "cmd": f"python3 {script_path} 'https://www.youtube.com/watch?v=EXAMPLE' -o ./test_4k_audio.webm --audio"
        },
        {
            "desc": "使用cookies下载",
            "cmd": f"python3 {script_path} 'https://www.youtube.com/watch?v=EXAMPLE' -o ./test_4k.webm -c cookies.txt"
        },
        {
            "desc": "设置超时和重试次数",
            "cmd": f"python3 {script_path} 'https://www.youtube.com/watch?v=EXAMPLE' -o ./test_4k.webm -t 3600 -r 5"
        }
    ]
    
    print("📝 使用示例：")
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}")
        print(f"   {example['cmd']}")
    
    print("\n" + "="*60)
    print("📋 参数说明：")
    print("   url         - YouTube视频链接（必需）")
    print("   -o, --output - 输出文件路径（默认：./video_4k.webm）")
    print("   -c, --cookies - cookies文件路径（可选）")
    print("   -a, --audio  - 包含音频轨道")
    print("   -t, --timeout - 下载超时时间（秒，默认1800）")
    print("   -r, --retries - 重试次数（默认3）")
    
    print("\n🔧 优化特性：")
    print("   ✅ 专门针对4K WebM格式优化")
    print("   ✅ 支持VP9和AV1编码")
    print("   ✅ 保持原始WebM容器格式")
    print("   ✅ 支持断点续传")
    print("   ✅ 多重格式回退机制")
    print("   ✅ 详细的视频信息验证")
    
    print("\n💡 使用建议：")
    print("   1. 优先使用VP9编码的WebM格式（更好兼容性）")
    print("   2. AV1编码提供更好压缩率但需要新版解码器")
    print("   3. 如果需要最大兼容性，可以用ffmpeg转换为MP4")
    print("   4. 大文件下载建议增加超时时间")

def interactive_download():
    """交互式下载"""
    print("\n🎯 交互式下载模式")
    
    url = input("请输入YouTube视频链接: ").strip()
    if not url:
        print("❌ 必须输入有效的视频链接")
        return
    
    output = input("输出文件名（默认：video_4k.webm）: ").strip()
    if not output:
        output = "video_4k.webm"
    
    audio = input("是否包含音频？(y/N): ").strip().lower() == 'y'
    
    script_path = Path(__file__).parent / "download_4k_webm.py"
    cmd = [
        "python3", str(script_path),
        url,
        "-o", output
    ]
    
    if audio:
        cmd.append("--audio")
    
    # 检查cookies文件
    cookies_file = Path("cookies.txt")
    if cookies_file.exists():
        use_cookies = input("发现cookies.txt文件，是否使用？(Y/n): ").strip().lower()
        if use_cookies != 'n':
            cmd.extend(["-c", str(cookies_file)])
    
    print(f"\n🚀 执行命令：{' '.join(cmd)}")
    print("开始下载...")
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ 下载完成！")
        else:
            print("❌ 下载失败")
    except KeyboardInterrupt:
        print("\n⚠️ 用户取消下载")
    except Exception as e:
        print(f"❌ 执行错误：{e}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_download()
    else:
        example_usage()
        
        choice = input("\n是否启动交互式下载？(y/N): ").strip().lower()
        if choice == 'y':
            interactive_download()

if __name__ == "__main__":
    main()
