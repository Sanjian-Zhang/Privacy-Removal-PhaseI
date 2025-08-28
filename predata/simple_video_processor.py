#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的视频处理命令行工具
用法: python simple_video_processor.py <视频URL> <视频编号>
"""

import os
import sys
import argparse

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from video_processing_pipeline import VideoPipelineProcessor

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="简单的视频处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python simple_video_processor.py "https://www.youtube.com/watch?v=RDLgS3vZlsE" "25"
  python simple_video_processor.py "https://www.youtube.com/watch?v=VIDEO_ID" "26" --fps 5
  python simple_video_processor.py "https://www.youtube.com/watch?v=VIDEO_ID" "27" --with_audio
        """
    )
    
    # 必需参数
    parser.add_argument("url", help="视频下载URL")
    parser.add_argument("video_number", help="视频编号")
    
    # 可选参数
    parser.add_argument("--fps", type=float, default=3.0, 
                      help="抽帧帧率 (默认: 3.0)")
    parser.add_argument("--with_audio", action="store_true",
                      help="下载时包含音频")
    parser.add_argument("--height", type=int, default=2160,
                      help="视频高度限制 (默认: 2160)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                      help="图片格式 (默认: jpg)")
    parser.add_argument("--start", type=float,
                      help="抽帧起始时间(秒)")
    parser.add_argument("--duration", type=float,
                      help="抽帧时长(秒)")
    
    args = parser.parse_args()
    
    try:
        print("🚀 启动视频处理流水线...")
        print(f"📹 视频URL: {args.url}")
        print(f"🏷️  视频编号: {args.video_number}")
        
        # 创建处理器
        processor = VideoPipelineProcessor()
        
        # 准备参数
        kwargs = {
            'cookies': 'cookies.txt',
            'with_audio': args.with_audio,
            'height': args.height,
            'fps': args.fps,
            'format': args.format,
            'start': args.start,
            'duration': args.duration,
            'timeout': 1200,
            'retries': 2,
            'jpg_q': 1
        }
        
        # 执行处理
        result = processor.process_video(args.url, args.video_number, **kwargs)
        
        # 保存日志
        processor.save_processing_log()
        
        # 检查结果
        if result['status'] == 'success':
            print("\n🎉 视频处理完成！")
            print(f"✅ 总耗时: {result['total_duration']:.1f}秒")
            
            # 显示输出位置
            if 'extract' in result['steps'] and result['steps']['extract']['frames_dir']:
                frames_dir = result['steps']['extract']['frames_dir']
                classified_dir = os.path.join(frames_dir, f"classified_frames{args.video_number}")
                print(f"📁 抽帧图片: {frames_dir}")
                print(f"📁 分类结果: {classified_dir}")
            
            return True
        else:
            print("\n❌ 视频处理失败！")
            print(f"❌ 失败步骤: {result.get('failure_step', '未知')}")
            if 'error' in result:
                print(f"❌ 错误信息: {result['error']}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚡ 用户中断操作")
        return False
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
