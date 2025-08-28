#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的视频处理器 - 主要优化点：
1. 更好的错误处理和日志记录
2. 配置管理系统
3. 内存管理优化
4. cuDNN兼容性修复
5. 更健壮的文件处理
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
import subprocess
import time

# 配置类
@dataclass
class VideoProcessingConfig:
    """视频处理配置"""
    # 视频处理
    fps: float = 3.0
    start_time: Optional[float] = None
    duration: Optional[float] = None
    scale_width: Optional[int] = None
    img_format: str = "jpg"
    jpg_quality: int = 1
    threads: int = 0
    
    # 下载配置
    max_height: int = 2160
    download_timeout: int = 1200
    download_retries: int = 3
    
    # GPU配置
    use_gpu: bool = True
    gpu_device_id: int = 0
    
    # 错误处理
    continue_on_error: bool = False
    max_retry_attempts: int = 3

class VideoProcessorError(Exception):
    """自定义视频处理异常"""
    pass

class OptimizedVideoProcessor:
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_environment()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("VideoProcessor")
        logger.setLevel(logging.INFO)
        
        # 如果logger已经有handler，不重复添加
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_environment(self):
        """设置环境变量和GPU配置"""
        try:
            # 修复cuDNN版本不匹配问题
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_device_id)
            
            # 设置PyTorch相关环境变量
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            if self.config.use_gpu:
                self._check_gpu_availability()
                
        except Exception as e:
            self.logger.warning(f"环境设置警告: {e}")
    
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if self.config.gpu_device_id >= device_count:
                    raise VideoProcessorError(
                        f"指定的GPU设备ID {self.config.gpu_device_id} 超出范围 (0-{device_count-1})"
                    )
                
                device_name = torch.cuda.get_device_name(self.config.gpu_device_id)
                memory_gb = torch.cuda.get_device_properties(self.config.gpu_device_id).total_memory / 1e9
                self.logger.info(f"🚀 使用GPU: {device_name} ({memory_gb:.1f}GB)")
            else:
                self.logger.warning("⚠️ CUDA不可用，将使用CPU")
                self.config.use_gpu = False
        except ImportError:
            self.logger.warning("⚠️ PyTorch未安装，无法检查GPU")
            self.config.use_gpu = False
    
    @contextmanager
    def _safe_temp_dir(self, prefix="video_proc_"):
        """安全的临时目录管理"""
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
            yield temp_dir
        finally:
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _run_command_with_retry(self, cmd: List[str], max_retries: Optional[int] = None) -> subprocess.CompletedProcess:
        """带重试机制的命令执行"""
        if max_retries is None:
            max_retries = self.config.max_retry_attempts
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"🚀 执行命令 (尝试 {attempt + 1}/{max_retries + 1}): {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.config.download_timeout
                )
                
                if result.returncode == 0:
                    return result
                else:
                    raise VideoProcessorError(f"命令执行失败: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                last_error = "命令执行超时"
                self.logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {last_error}")
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {last_error}")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 指数退避
                self.logger.info(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        raise VideoProcessorError(f"命令执行失败，已尝试 {max_retries + 1} 次: {last_error}")
    
    def check_video_integrity(self, video_path: Path) -> bool:
        """检查视频文件完整性"""
        try:
            if not video_path.exists():
                return False
            
            if video_path.stat().st_size == 0:
                return False
            
            # 使用ffprobe检查文件头
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0",
                "-show_entries", "format=format_name,duration", 
                "-of", "json", 
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.logger.error(f"ffprobe检查失败: {result.stderr}")
                return False
            
            try:
                info = json.loads(result.stdout)
                if 'format' not in info:
                    return False
                
                duration = float(info['format'].get('duration', 0))
                if duration <= 0:
                    self.logger.warning(f"视频时长异常: {duration}")
                    return False
                
                self.logger.info(f"✅ 视频检查通过: 时长 {duration:.1f}s")
                return True
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.error(f"解析ffprobe输出失败: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"视频完整性检查失败: {e}")
            return False
    
    def extract_frames_safe(self, video_path: Path, output_dir: Path) -> Tuple[bool, int]:
        """安全的帧抽取，返回(成功状态, 生成帧数)"""
        try:
            # 检查视频完整性
            if not self.check_video_integrity(video_path):
                self.logger.error(f"视频文件损坏或不可读: {video_path}")
                return False, 0
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成输出文件名模式
            video_name = video_path.stem
            import re
            match = re.search(r'(\d+)', video_name)
            video_prefix = f"video{match.group(1)}" if match else video_name
            
            # 构建ffmpeg命令
            cmd = self._build_ffmpeg_command(video_path, output_dir, video_prefix)
            
            # 执行抽帧
            result = self._run_command_with_retry(cmd)
            
            # 统计生成的帧数
            frame_count = len([f for f in output_dir.glob(f"{video_prefix}_frame_*.{self.config.img_format.lower()}")])
            
            if frame_count > 0:
                self.logger.info(f"✅ 抽帧完成: {frame_count} 帧")
                return True, frame_count
            else:
                self.logger.error("❌ 未生成任何帧")
                return False, 0
                
        except Exception as e:
            self.logger.error(f"帧抽取失败: {e}")
            return False, 0
    
    def _build_ffmpeg_command(self, video_path: Path, output_dir: Path, video_prefix: str) -> List[str]:
        """构建ffmpeg命令"""
        pattern = output_dir / f"{video_prefix}_frame_%06d.{self.config.img_format.lower()}"
        
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
        
        # 添加起始时间
        if self.config.start_time is not None:
            cmd.extend(["-ss", str(self.config.start_time)])
        
        # 输入文件
        cmd.extend(["-i", str(video_path)])
        
        # 视频滤镜
        vf_filters = [f"fps={self.config.fps}"]
        if self.config.scale_width:
            vf_filters.append(f"scale={self.config.scale_width}:-2:flags=bicubic")
        
        cmd.extend([
            "-vf", ",".join(vf_filters),
            "-vsync", "vfr",
            "-threads", str(self.config.threads),
            "-map_metadata", "-1",
            "-an"  # 不处理音频
        ])
        
        # 添加时长限制
        if self.config.duration is not None:
            cmd.extend(["-t", str(self.config.duration)])
        
        # 输出格式参数
        if self.config.img_format.lower() in ("jpg", "jpeg"):
            cmd.extend(["-q:v", str(self.config.jpg_quality), "-vcodec", "mjpeg"])
        elif self.config.img_format.lower() == "png":
            cmd.extend(["-compression_level", "4"])
        
        cmd.extend(["-y", str(pattern)])
        return cmd
    
    def process_video_safe(self, video_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """安全的视频处理入口"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        result = {
            "success": False,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "frame_count": 0,
            "error": None,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🎬 开始处理视频: {video_path}")
            
            if not video_path.exists():
                raise VideoProcessorError(f"视频文件不存在: {video_path}")
            
            success, frame_count = self.extract_frames_safe(video_path, output_dir)
            
            result.update({
                "success": success,
                "frame_count": frame_count,
                "processing_time": time.time() - start_time
            })
            
            if success:
                self.logger.info(f"🎉 视频处理完成: {frame_count} 帧，耗时 {result['processing_time']:.1f}s")
            else:
                result["error"] = "帧抽取失败"
                
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ 视频处理失败: {e}")
        finally:
            result["processing_time"] = time.time() - start_time
        
        return result

# 使用示例
def main():
    # 创建配置
    config = VideoProcessingConfig(
        fps=1.0,
        img_format="jpg",
        jpg_quality=1,
        use_gpu=True,
        continue_on_error=True
    )
    
    # 创建处理器
    processor = OptimizedVideoProcessor(config)
    
    # 处理视频
    video_path = "/home/zhiqics/sanjian/predata/downloaded_video40.mp4"
    output_dir = "/home/zhiqics/sanjian/predata/output_frames40_optimized"
    
    result = processor.process_video_safe(video_path, output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
