#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速人脸车牌检测器 - 高度优化版本
专注于速度和稳定性，移除复杂的预过滤逻辑
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import multiprocessing as mp
import math
import re
import threading
import psutil
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager

# GPU加速相关导入
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy可用，将使用GPU加速")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy不可用，将使用CPU处理")

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 修复CUDA多进程问题
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fast_face_plate_detector_v2.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查并导入依赖库"""
    missing_deps = []
    imported_modules = {}
    
    try:
        from retinaface import RetinaFace
        imported_modules['RetinaFace'] = RetinaFace
        logger.info("✅ RetinaFace 库导入成功")
    except ImportError as e:
        logger.error(f"❌ RetinaFace 库导入失败: {e}")
        missing_deps.append("retina-face")
    
    try:
        from ultralytics import YOLO
        imported_modules['YOLO'] = YOLO
        logger.info("✅ YOLO 库导入成功")
    except ImportError as e:
        logger.error(f"❌ YOLO 库导入失败: {e}")
        missing_deps.append("ultralytics")
    
    try:
        import easyocr
        imported_modules['easyocr'] = easyocr
        logger.info("✅ EasyOCR 库导入成功")
    except ImportError as e:
        logger.error(f"❌ EasyOCR 库导入失败: {e}")
        missing_deps.append("easyocr")
    
    try:
        import torch
        imported_modules['torch'] = torch
        logger.info("✅ PyTorch 库导入成功")
    except ImportError as e:
        logger.error(f"❌ PyTorch 库导入失败: {e}")
        missing_deps.append("torch")
    
    if missing_deps:
        logger.error(f"❌ 缺少依赖库: {', '.join(missing_deps)}")
        return None
    
    return imported_modules

# 检查并导入依赖
modules = check_dependencies()
if modules is None:
    exit(1)

RetinaFace = modules['RetinaFace']
YOLO = modules['YOLO']
easyocr = modules['easyocr']
torch = modules['torch']

class GPUMemoryManager:
    """GPU内存管理器 - 优化版"""
    
    def __init__(self, max_gpu_memory_mb=3072, max_cpu_memory_mb=1536, warning_threshold=0.7):
        """
        初始化GPU内存管理器
        
        Args:
            max_gpu_memory_mb (int): 最大GPU内存使用量(MB) - 降低默认值
            max_cpu_memory_mb (int): 最大CPU内存使用量(MB) - 降低默认值
            warning_threshold (float): 内存警告阈值(0-1) - 降低阈值
        """
        self.max_gpu_memory_bytes = max_gpu_memory_mb * 1024 * 1024
        self.max_cpu_memory_bytes = max_cpu_memory_mb * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_available = self._check_gpu_availability()
        self.cleanup_counter = 0  # 清理计数器
        self.force_cleanup_threshold = 50  # 强制清理阈值
        
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        if not CUPY_AVAILABLE:
            return False
            
        try:
            cp.cuda.Device(0).use()
            return True
        except Exception as e:
            print(f"⚠️ GPU不可用: {str(e)}")
            return False
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if not self.gpu_available:
            return {'used': 0, 'free': 0, 'total': 0}
            
        try:
            mempool = cp.get_default_memory_pool()
            return {
                'used': mempool.used_bytes(),
                'total': mempool.total_bytes(),
                'free': mempool.total_bytes() - mempool.used_bytes()
            }
        except Exception:
            return {'used': 0, 'free': 0, 'total': 0}
    
    def get_cpu_memory_usage(self):
        """获取CPU内存使用情况"""
        try:
            memory_info = self.process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': self.process.memory_percent(),
                'available': psutil.virtual_memory().available
            }
        except Exception:
            return {'rss': 0, 'vms': 0, 'percent': 0, 'available': 0}
    
    def is_gpu_memory_available(self, required_bytes=0):
        """检查GPU是否有足够内存"""
        if not self.gpu_available:
            return False
            
        gpu_memory = self.get_gpu_memory_usage()
        return (gpu_memory['used'] + required_bytes) < self.max_gpu_memory_bytes
    
    def force_gpu_cleanup(self):
        """强制GPU内存清理"""
        if self.gpu_available and CUPY_AVAILABLE:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass
        gc.collect()
        self.cleanup_counter += 1
    
    def start_monitoring(self):
        """开始内存监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_memory(self):
        """内存监控线程 - 优化版"""
        while self.monitoring:
            cpu_memory = self.get_cpu_memory_usage()
            
            # CPU内存监控 - 更严格的控制
            cpu_usage_ratio = cpu_memory['rss'] / self.max_cpu_memory_bytes
            if cpu_usage_ratio > self.warning_threshold:
                logger.warning(f"⚠️ CPU内存使用率过高: {cpu_usage_ratio:.1%}")
                # 执行强制内存清理
                self.force_gpu_cleanup()
                
                # 如果内存使用率超过90%，触发紧急清理
                if cpu_usage_ratio > 0.9:
                    logger.error(f"❌ CPU内存使用率危险: {cpu_usage_ratio:.1%}")
                    # 强制Python垃圾回收
                    for _ in range(3):
                        gc.collect()
            
            # GPU内存监控
            if self.gpu_available and CUPY_AVAILABLE:
                gpu_memory = self.get_gpu_memory_usage()
                if gpu_memory['total'] > 0:
                    gpu_usage_ratio = gpu_memory['used'] / gpu_memory['total']
                    if gpu_usage_ratio > self.warning_threshold:
                        logger.warning(f"⚠️ GPU内存使用率过高: {gpu_usage_ratio:.1%}")
                        self.force_gpu_cleanup()
            
            time.sleep(2)  # 更频繁的监控：每2秒检查一次
    
    def get_dynamic_batch_size(self, default_batch_size=32):
        """根据内存使用情况动态调整批处理大小"""
        cpu_memory = self.get_cpu_memory_usage()
        cpu_usage_ratio = cpu_memory['rss'] / self.max_cpu_memory_bytes
        
        if cpu_usage_ratio > 0.8:
            return max(8, default_batch_size // 4)  # 内存紧张时减少到1/4
        elif cpu_usage_ratio > 0.6:
            return max(16, default_batch_size // 2)  # 内存较紧张时减少到1/2
        else:
            return default_batch_size


class SimilarImageDetectorGPU:
    """GPU加速的相似图片检测器"""
    
    def __init__(self, psnr_threshold=50.0, max_gpu_memory_mb=3072, max_cpu_memory_mb=1536, 
                 use_gpu=True, min_frame_distance=5, adjacent_frame_threshold=8):
        """
        初始化GPU加速的相似图片检测器 - 内存优化版
        
        Args:
            psnr_threshold (float): PSNR阈值
            max_gpu_memory_mb (int): 最大GPU内存使用量(MB) - 降低默认值
            max_cpu_memory_mb (int): 最大CPU内存使用量(MB) - 降低默认值
            use_gpu (bool): 是否使用GPU加速
            min_frame_distance (int): 最小帧间距，用于过滤相邻帧
            adjacent_frame_threshold (int): 相邻帧相似阈值，≤此值认定为相似
        """
        self.psnr_threshold = psnr_threshold
        self.min_frame_distance = min_frame_distance
        self.adjacent_frame_threshold = adjacent_frame_threshold
        self.memory_manager = GPUMemoryManager(max_gpu_memory_mb, max_cpu_memory_mb)
        self.use_gpu = use_gpu and self.memory_manager.gpu_available
        
        # 初始化CUDA上下文（如果使用GPU）
        if self.use_gpu:
            try:
                if CUPY_AVAILABLE:
                    import cupy as cp
                    cp.cuda.Device(0).use()
                logger.info("🔧 GPU初始化成功，将使用GPU加速相似度检测")
            except Exception as e:
                logger.warning(f"⚠️ GPU初始化失败，使用CPU: {e}")
                self.use_gpu = False
        
        if not self.use_gpu:
            logger.info("💻 使用CPU处理模式进行相似度检测")
        
        self.stats = {
            'total_images': 0,
            'similar_groups': 0,
            'similar_images': 0,
            'unique_images': 0,
            'processed_pairs': 0,
            'memory_cleanups': 0,
            'max_memory_used': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'adjacent_frames_filtered': 0,
            'adjacent_frames_similar': 0
        }
    
    def extract_frame_number(self, filename):
        """从文件名中提取帧号"""
        patterns = [
            r'frame_(\d+)',           # frame_001.jpg
            r'frame(\d+)',            # frame001.jpg  
            r'_(\d+)\.jpg',           # xxx_001.jpg
            r'_(\d+)\.png',           # xxx_001.png
            r'(\d+)\.jpg',            # 001.jpg
            r'(\d+)\.png',            # 001.png
            r'-(\d+)\.',              # xxx-001.jpg
            r'(\d{3,})',              # 任何3位以上数字
        ]
        
        filename_str = str(filename)
        for pattern in patterns:
            match = re.search(pattern, filename_str)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return -1  # 无法提取帧号
    
    def calculate_image_clarity(self, image_path):
        """计算图片的清晰度分数 - 内存优化版"""
        try:
            if not Path(image_path).exists():
                return 0.0
            
            # 使用更小的图片尺寸来节省内存
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0
            
            if img.shape[0] == 0 or img.shape[1] == 0:
                return 0.0
            
            # 如果图片太大，先缩放以节省内存
            height, width = img.shape[:2]
            if height > 800 or width > 800:
                scale_factor = min(800/height, 800/width)
                new_height, new_width = int(height * scale_factor), int(width * scale_factor)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 立即释放彩色图像内存
            del img
            
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    import cupy as cp
                    gray_gpu = cp.asarray(gray)
                    laplacian_var = cp.var(cp.array([
                        [-1, -2, -1], [0, 0, 0], [1, 2, 1]
                    ], dtype=cp.float32))
                    quality_score = float(laplacian_var)
                    
                    # 立即释放GPU内存
                    del gray_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    self.stats['gpu_operations'] += 1
                except:
                    quality_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    self.stats['cpu_operations'] += 1
            else:
                quality_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                self.stats['cpu_operations'] += 1
            
            # 立即释放灰度图像内存
            del gray
            
            # 结合文件大小作为质量指标
            try:
                file_size = os.path.getsize(image_path)
                size_factor = min(file_size / (500 * 1024), 1.0)  # 500KB为基准
                quality_score = quality_score * size_factor
            except OSError:
                pass
            
            return quality_score
            
        except Exception as e:
            logger.debug(f"⚠️ 清晰度计算失败 {Path(image_path).name}: {str(e)}")
            return 0.0
        finally:
            # 强制垃圾回收
            gc.collect()
    
    def calculate_psnr_gpu(self, img1, img2):
        """GPU加速的PSNR计算 - 内存优化版"""
        try:
            # 处理不同尺寸的图片
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            min_h = min(h1, h2)
            min_w = min(w1, w2)
            
            # 限制最大处理尺寸以节省内存
            max_size = 512  # 最大处理尺寸
            if min_h > max_size or min_w > max_size:
                scale_factor = min(max_size/min_h, max_size/min_w)
                min_h = int(min_h * scale_factor)
                min_w = int(min_w * scale_factor)
            
            def center_crop_and_resize(img, target_h, target_w):
                h, w = img.shape[:2]
                start_h = (h - min(h, target_h * 2)) // 2  # 取中心区域
                start_w = (w - min(w, target_w * 2)) // 2
                end_h = start_h + min(h, target_h * 2)
                end_w = start_w + min(w, target_w * 2)
                
                cropped = img[start_h:end_h, start_w:end_w]
                return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            img1_cropped = center_crop_and_resize(img1, min_h, min_w)
            img2_cropped = center_crop_and_resize(img2, min_h, min_w)
            
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    import cupy as cp
                    img1_gpu = cp.asarray(img1_cropped, dtype=cp.float64)
                    img2_gpu = cp.asarray(img2_cropped, dtype=cp.float64)
                    mse_gpu = cp.mean((img1_gpu - img2_gpu) ** 2)
                    mse_cpu = float(mse_gpu)
                    
                    # 立即释放GPU内存
                    del img1_gpu, img2_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    self.stats['gpu_operations'] += 1
                except:
                    mse_cpu = np.mean((img1_cropped.astype(np.float64) - img2_cropped.astype(np.float64)) ** 2)
                    self.stats['cpu_operations'] += 1
            else:
                mse_cpu = np.mean((img1_cropped.astype(np.float64) - img2_cropped.astype(np.float64)) ** 2)
                self.stats['cpu_operations'] += 1
            
            # 立即释放处理后的图像内存
            del img1_cropped, img2_cropped
            
            if mse_cpu == 0:
                return 100.0  # 完全相同
            
            # 计算PSNR
            max_pixel_value = 255.0
            psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse_cpu))
            
            return psnr
            
        except Exception as e:
            logger.debug(f"⚠️ GPU PSNR计算失败，使用CPU: {str(e)}")
            return self.calculate_psnr_cpu_fallback(img1, img2)
        finally:
            # GPU内存清理
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            # 强制垃圾回收
            gc.collect()
    
    def calculate_psnr_cpu_fallback(self, img1, img2):
        """CPU版本的PSNR计算（作为GPU失败时的回退）"""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            min_h = min(h1, h2)
            min_w = min(w1, w2)
            
            def center_crop(img, target_h, target_w):
                h, w = img.shape[:2]
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                return img[start_h:start_h + target_h, start_w:start_w + target_w]
            
            if h1 != h2 or w1 != w2:
                img1_cropped = center_crop(img1, min_h, min_w)
                img2_cropped = center_crop(img2, min_h, min_w)
            else:
                img1_cropped = img1
                img2_cropped = img2
            
            mse = np.mean((img1_cropped.astype(np.float64) - img2_cropped.astype(np.float64)) ** 2)
            
            if mse == 0:
                return 100.0
            
            max_pixel_value = 255.0
            psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
            
            self.stats['cpu_operations'] += 1
            return psnr
            
        except Exception as e:
            logger.debug(f"⚠️ CPU PSNR计算出错: {str(e)}")
            return 0.0
    
    def are_images_similar(self, image_path1, image_path2, check_adjacent_frames=True, adjacent_frame_threshold=8):
        """判断两张图片是否相似（GPU加速版）"""
        img1 = None
        img2 = None
        try:
            # 优先检查相邻帧：如果帧间距≤8，直接认定为相似
            if check_adjacent_frames:
                frame1 = self.extract_frame_number(image_path1.name if hasattr(image_path1, 'name') else os.path.basename(image_path1))
                frame2 = self.extract_frame_number(image_path2.name if hasattr(image_path2, 'name') else os.path.basename(image_path2))
                
                if frame1 != -1 and frame2 != -1:
                    frame_distance = abs(frame1 - frame2)
                    if frame_distance <= adjacent_frame_threshold:
                        self.stats['adjacent_frames_similar'] += 1
                        
                        clarity1 = self.calculate_image_clarity(image_path1)
                        clarity2 = self.calculate_image_clarity(image_path2)
                        
                        details = {
                            'frame1': frame1,
                            'frame2': frame2,
                            'frame_distance': frame_distance,
                            'clarity1': clarity1,
                            'clarity2': clarity2,
                            'reason': 'adjacent_frames_similar'
                        }
                        
                        return True, 95.0, details
            
            # 检查内存
            if not self.memory_manager.is_gpu_memory_available():
                self.memory_manager.force_gpu_cleanup()
            
            # 读取图片
            try:
                img1 = cv2.imread(str(image_path1), cv2.IMREAD_COLOR)
                if img1 is None:
                    return False, 0.0, {'error': f'无法读取图片1: {image_path1}'}
            except Exception as e:
                return False, 0.0, {'error': f'图片1读取异常: {str(e)}'}
            
            try:
                img2 = cv2.imread(str(image_path2), cv2.IMREAD_COLOR)
                if img2 is None:
                    return False, 0.0, {'error': f'无法读取图片2: {image_path2}'}
            except Exception as e:
                return False, 0.0, {'error': f'图片2读取异常: {str(e)}'}
            
            # 检查图片尺寸
            if img1.shape == (0, 0, 3) or img2.shape == (0, 0, 3):
                return False, 0.0, {'error': '图片尺寸无效'}
            
            # GPU加速的相似度计算
            psnr = self.calculate_psnr_gpu(img1, img2)
            
            # 相似性判断
            is_similar = psnr > self.psnr_threshold
            
            details = {
                'psnr': psnr,
                'psnr_threshold': self.psnr_threshold,
                'gpu_used': self.use_gpu
            }
            
            return is_similar, psnr, details
            
        except Exception as e:
            logger.debug(f"⚠️ GPU相似度检测出错: {str(e)}")
            return False, 0.0, {'error': str(e)}
        finally:
            # 及时释放内存
            if img1 is not None:
                del img1
            if img2 is not None:
                del img2
            
            # GPU内存清理
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            gc.collect()
    
    def find_similar_groups(self, image_paths, progress_callback=None):
        """GPU加速的相似图片组检测"""
        self.memory_manager.start_monitoring()
        
        try:
            similar_groups = []
            total_comparisons = len(image_paths) - 1
            current_comparison = 0
            
            logger.info(f"🚀 开始GPU加速相似图片检测...")
            logger.info(f"   - 图片总数: {len(image_paths)}")
            logger.info(f"   - PSNR阈值: {self.psnr_threshold} dB")
            logger.info(f"   - GPU加速: {'启用' if self.use_gpu else '禁用'}")
            
            if len(image_paths) == 0:
                return similar_groups
            
            i = 0
            while i < len(image_paths):
                current_group = [image_paths[i]]
                
                j = i + 1
                while j < len(image_paths):
                    is_similar, score, details = self.are_images_similar(
                        image_paths[i], image_paths[j], 
                        check_adjacent_frames=True, 
                        adjacent_frame_threshold=self.adjacent_frame_threshold
                    )
                    
                    if is_similar:
                        current_group.append(image_paths[j])
                        image_paths.pop(j)
                    else:
                        j += 1
                    
                    current_comparison += 1
                    
                    if progress_callback and current_comparison % 50 == 0:
                        progress_callback(current_comparison, total_comparisons)
                
                similar_groups.append(current_group)
                i += 1
            
            # 更新统计信息
            self.stats['total_images'] = sum(len(g) for g in similar_groups)
            self.stats['similar_groups'] = len([g for g in similar_groups if len(g) > 1])
            self.stats['similar_images'] = sum(len(g) - 1 for g in similar_groups if len(g) > 1)
            self.stats['unique_images'] = len(similar_groups)
            
            logger.info(f"📊 GPU加速检测完成:")
            logger.info(f"   - 发现 {self.stats['similar_groups']} 个相似序列")
            logger.info(f"   - 保留 {self.stats['unique_images']} 张代表图片")
            logger.info(f"   - 过滤 {self.stats['similar_images']} 张冗余图片")
            
            return similar_groups
            
        finally:
            self.memory_manager.stop_monitoring()
    
    def auto_deduplicate(self, image_paths, output_dir, similar_dir):
        """GPU加速的自动去重"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(similar_dir, exist_ok=True)
        
        similar_groups = self.find_similar_groups(image_paths)
        
        logger.info(f"\n🚀 开始GPU加速自动去重...")
        
        unique_images = []
        similar_pairs = []
        
        for group_idx, group in enumerate(similar_groups, 1):
            if len(group) == 1:
                unique_images.append(group[0])
            else:
                # 选择清晰度最高的图片作为代表
                best_image = group[0]
                best_clarity = self.calculate_image_clarity(group[0])
                
                for img_path in group[1:]:
                    clarity = self.calculate_image_clarity(img_path)
                    if clarity > best_clarity:
                        best_clarity = clarity
                        best_image = img_path
                
                unique_images.append(best_image)
                
                # 记录相似图片对
                for img_path in group:
                    if img_path != best_image:
                        similar_pairs.append((best_image, img_path))
        
        # 移动唯一图片
        logger.info(f"\n📋 移动唯一图片到输出目录...")
        successful_moves = 0
        for i, unique_img in enumerate(unique_images, 1):
            try:
                filename = os.path.basename(unique_img)
                dst_path = os.path.join(output_dir, filename)
                
                # 处理文件名冲突
                counter = 1
                while os.path.exists(dst_path):
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{name}_{counter}{ext}"
                    dst_path = os.path.join(output_dir, new_filename)
                    counter += 1
                
                shutil.copy2(unique_img, dst_path)
                successful_moves += 1
                
                if i % 100 == 0:
                    logger.info(f"   进度: {i}/{len(unique_images)}")
                    
            except Exception as e:
                logger.error(f"移动图片失败 {unique_img}: {e}")
        
        # 移动相似图片
        logger.info(f"\n📋 移动相似图片到相似目录...")
        for best_img, similar_img in similar_pairs:
            try:
                filename = os.path.basename(similar_img)
                dst_path = os.path.join(similar_dir, filename)
                
                counter = 1
                while os.path.exists(dst_path):
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{name}_{counter}{ext}"
                    dst_path = os.path.join(similar_dir, new_filename)
                    counter += 1
                
                shutil.copy2(similar_img, dst_path)
                
            except Exception as e:
                logger.error(f"移动相似图片失败 {similar_img}: {e}")
        
        logger.info(f"   ✅ 成功移动: {successful_moves}/{len(unique_images)} 张图片")
        
        # GPU内存清理
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        return unique_images, similar_pairs


class FastConfig:
    """快速检测配置类"""
    
    def __init__(self, input_dir=None):
        """
        初始化配置
        
        Args:
            input_dir: 输入目录路径，如果为None则使用默认值
        """
        # 目录配置
        if input_dir:
            self.INPUT_DIR = input_dir
            # 输出目录设置在输入目录的下一级
            self.OUTPUT_BASE_DIR = os.path.join(input_dir, "processed_output")
        else:
            self.INPUT_DIR = '/home/zhiqics/sanjian/predata/output_frames70'
            self.OUTPUT_BASE_DIR = '/home/zhiqics/sanjian/predata/output_frames70'

        # 模型路径
        self.YOLOV8S_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/yolov8s.pt'
        self.LICENSE_PLATE_MODEL_PATH = '/home/zhiqics/sanjian/predata/models/license_plate_detector.pt'
        
        # 检测阈值
        self.MIN_FACE_CONFIDENCE_RETINA = 0.85     # 提高RetinaFace置信度阈值
        self.MIN_PLATE_CONFIDENCE = 0.8
        self.MIN_TEXT_CONFIDENCE = 0.5
        self.YAW_ANGLE_THRESHOLD = 18.0           # 降低yaw角度阈值，更严格
        
        # 评分系统
        self.SCORE_PER_CLEAR_FRONTAL_FACE = 2
        self.SCORE_PER_CLEAR_PLATE = 2
        self.SCORE_PER_TEXT = 1
        self.REQUIRED_TOTAL_SCORE = 5
        
        # 近景判断参数 - 更严格地过滤远处人脸和后脑勺
        self.MIN_FACE_SIZE = 140                   # 进一步提高最小人脸尺寸
        self.CLOSE_UP_FACE_RATIO = 0.15            # 提高面积比例阈值
        self.MIN_FACE_AREA = 19600                 # 提高最小人脸面积 (140x140)
        self.MAX_DISTANCE_THRESHOLD = 0.55         # 降低边缘距离阈值，更严格
        self.MIN_FACE_RESOLUTION = 160             # 提高最小分辨率要求
        
        # 相似图片检测配置 - 内存优化
        self.ENABLE_SIMILARITY_DETECTION = True     # 是否启用相似图片检测
        self.PSNR_THRESHOLD = 50.0                 # PSNR相似度阈值
        self.ADJACENT_FRAME_THRESHOLD = 8          # 相邻帧相似阈值
        self.MIN_FRAME_DISTANCE = 5                # 最小帧间距
        self.MAX_GPU_MEMORY_MB = 3072              # 最大GPU内存使用量(MB) - 降低
        self.MAX_CPU_MEMORY_MB = 1536              # 最大CPU内存使用量(MB) - 降低
        
        # 文件格式
        self.SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 优化参数（专注速度和内存效率）
        self.BATCH_SIZE = 16                       # 降低批处理大小
        self.ENABLE_PREFILTER = False              # 禁用预过滤
        self.PROGRESS_UPDATE_FREQUENCY = 50        # 更频繁的进度更新
        self.MEMORY_CLEANUP_INTERVAL = 20          # 每20个批次强制清理内存
    
    def get_output_dirs(self):
        """获取输出目录配置"""
        return {
            'high_score': os.path.join(self.OUTPUT_BASE_DIR, "high_score_images"),
            'low_score': os.path.join(self.OUTPUT_BASE_DIR, "low_score_images"),
            'zero_score': os.path.join(self.OUTPUT_BASE_DIR, "zero_score_images"),
            'analysis': os.path.join(self.OUTPUT_BASE_DIR, "analysis"),
            'unique_high_score': os.path.join(self.OUTPUT_BASE_DIR, "unique_high_score_images"),  # 新增：去重后的高分图片
            'similar_high_score': os.path.join(self.OUTPUT_BASE_DIR, "similar_high_score_images"), # 新增：相似的高分图片
        }

class SimpleProgressBar:
    """简化的进度条"""
    
    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: Optional[int] = None):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        now = time.time()
        if now - self.last_update < 2.0 and self.current < self.total:  # 每2秒更新一次
            return
        
        self.last_update = now
        progress = self.current / self.total if self.total > 0 else 0
        percent = progress * 100
        
        elapsed = now - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
            logger.info(f"{self.prefix}: {self.current}/{self.total} ({percent:.1f}%) "
                       f"速度: {speed:.1f}/s 剩余: {eta:.0f}s")
        
        if self.current >= self.total:
            logger.info(f"✅ {self.prefix} 完成!")

class FastProcessor:
    """快速处理器"""
    
    def __init__(self, config: FastConfig):
        self.config = config
        self.device = None
        self.models = {}
        self.ocr_reader = None
        self.similarity_detector = None  # 新增：相似图片检测器
        self.stats = {
            'high_score': 0,
            'low_score': 0,
            'zero_score': 0,
            'failed': 0,
            'unique_high_score': 0,       # 新增：去重后的高分图片统计
            'similar_high_score': 0,      # 新增：相似的高分图片统计
        }
        
    def initialize_models(self):
        """初始化模型"""
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                self.device = 'cuda:0'
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.7, device=0)
                logger.info(f"🔧 GPU初始化成功: {self.device}")
            else:
                self.device = 'cpu'
                logger.warning("⚠️  使用CPU模式")
            
            logger.info("🔄 加载YOLO人脸模型...")
            self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
            
            logger.info("🔄 加载YOLO车牌模型...")
            self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
            
            logger.info("🔄 初始化OCR...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu='cuda' in self.device)
            
            # 初始化相似图片检测器
            if self.config.ENABLE_SIMILARITY_DETECTION:
                logger.info("🔄 初始化GPU加速相似图片检测器...")
                self.similarity_detector = SimilarImageDetectorGPU(
                    psnr_threshold=self.config.PSNR_THRESHOLD,
                    max_gpu_memory_mb=self.config.MAX_GPU_MEMORY_MB,
                    max_cpu_memory_mb=self.config.MAX_CPU_MEMORY_MB,
                    use_gpu='cuda' in self.device,
                    min_frame_distance=self.config.MIN_FRAME_DISTANCE,
                    adjacent_frame_threshold=self.config.ADJACENT_FRAME_THRESHOLD
                )
                logger.info("✅ 相似图片检测器初始化完成")
            
            logger.info(f"✅ 模型初始化完成 ({self.device})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            try:
                self.device = 'cpu'
                self.models['face'] = YOLO(self.config.YOLOV8S_MODEL_PATH)
                self.models['plate'] = YOLO(self.config.LICENSE_PLATE_MODEL_PATH)
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                
                # CPU模式下也初始化相似图片检测器
                if self.config.ENABLE_SIMILARITY_DETECTION:
                    self.similarity_detector = SimilarImageDetectorGPU(
                        psnr_threshold=self.config.PSNR_THRESHOLD,
                        max_gpu_memory_mb=self.config.MAX_GPU_MEMORY_MB,
                        max_cpu_memory_mb=self.config.MAX_CPU_MEMORY_MB,
                        use_gpu=False,
                        min_frame_distance=self.config.MIN_FRAME_DISTANCE,
                        adjacent_frame_threshold=self.config.ADJACENT_FRAME_THRESHOLD
                    )
                
                logger.info("✅ CPU模式初始化成功")
                return True
            except Exception as e2:
                logger.error(f"❌ CPU模式初始化也失败: {e2}")
                return False
    
    def calculate_yaw_angle(self, landmarks: Dict) -> float:
        """改进的yaw角度计算 - 更准确区分正脸和后脑勺"""
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            
            # 基础检查
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            eye_width = np.linalg.norm(eye_vector)
            
            if eye_width < 10:
                return 90.0
            
            # 改进的yaw计算
            horizontal_offset = nose[0] - eye_center[0]
            normalized_offset = horizontal_offset / eye_width
            
            # 更保守的角度计算，避免误判后脑勺
            yaw_angle = abs(normalized_offset) * 35.0  # 降低系数从60改为35
            
            # 额外的对称性检查
            if 'left_mouth_corner' in landmarks and 'right_mouth_corner' in landmarks:
                left_mouth = np.array(landmarks['left_mouth_corner'])
                right_mouth = np.array(landmarks['right_mouth_corner'])
                mouth_center = (left_mouth + right_mouth) / 2
                
                # 嘴部中心也应该在合理位置
                mouth_offset = mouth_center[0] - eye_center[0]
                mouth_normalized = mouth_offset / eye_width
                
                # 如果鼻子和嘴部偏移方向一致且都很大，可能是侧脸
                if abs(mouth_normalized) > 0.3 and np.sign(normalized_offset) == np.sign(mouth_normalized):
                    yaw_angle = min(yaw_angle * 1.5, 90.0)  # 增加角度惩罚
            
            return yaw_angle
        except Exception as e:
            logger.debug(f"Yaw角度计算失败: {e}")
            return 90.0
    
    def validate_frontal_face_features(self, landmarks: Dict, facial_area: List[int]) -> bool:
        """验证是否为真正的正脸特征（非后脑勺）"""
        try:
            x1, y1, x2, y2 = facial_area
            face_width = x2 - x1
            face_height = y2 - y1
            
            # 获取关键点
            left_eye = np.array(landmarks.get('left_eye', [0, 0]))
            right_eye = np.array(landmarks.get('right_eye', [0, 0]))
            nose = np.array(landmarks.get('nose', [0, 0]))
            left_mouth = np.array(landmarks.get('left_mouth_corner', [0, 0]))
            right_mouth = np.array(landmarks.get('right_mouth_corner', [0, 0]))
            
            # 检查关键点是否都存在且在合理位置
            required_points = [left_eye, right_eye, nose]
            for point in required_points:
                if np.allclose(point, [0, 0]):
                    return False  # 缺少关键点
                
                # 检查关键点是否在面部区域内
                if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                    return False
            
            # 1. 眼间距检查 - 正脸的眼间距应该合理
            eye_distance = np.linalg.norm(right_eye - left_eye)
            eye_distance_ratio = eye_distance / face_width
            
            # 正脸眼间距通常占面部宽度的25%-45%
            if not (0.20 <= eye_distance_ratio <= 0.50):
                logger.debug(f"眼间距异常: {eye_distance_ratio:.3f}")
                return False
            
            # 2. 面部对称性检查
            eye_center = (left_eye + right_eye) / 2
            
            # 鼻子应该在眼部中心线附近
            nose_offset = abs(nose[0] - eye_center[0])
            nose_offset_ratio = nose_offset / face_width
            
            if nose_offset_ratio > 0.15:  # 鼻子偏移不能超过面部宽度的15%
                logger.debug(f"鼻子偏移过大: {nose_offset_ratio:.3f}")
                return False
            
            # 3. 垂直位置检查
            if not np.allclose(left_mouth, [0, 0]) and not np.allclose(right_mouth, [0, 0]):
                mouth_center = (left_mouth + right_mouth) / 2
                
                # 眼部相对位置（应该在上半部分）
                eye_y_ratio = (eye_center[1] - y1) / face_height
                nose_y_ratio = (nose[1] - y1) / face_height
                mouth_y_ratio = (mouth_center[1] - y1) / face_height
                
                # 正脸的典型垂直分布
                if not (0.15 <= eye_y_ratio <= 0.45):
                    logger.debug(f"眼部垂直位置异常: {eye_y_ratio:.3f}")
                    return False
                
                if not (0.35 <= nose_y_ratio <= 0.70):
                    logger.debug(f"鼻子垂直位置异常: {nose_y_ratio:.3f}")
                    return False
                
                if not (0.60 <= mouth_y_ratio <= 0.90):
                    logger.debug(f"嘴部垂直位置异常: {mouth_y_ratio:.3f}")
                    return False
                
                # 嘴部也应该相对居中
                mouth_offset = abs(mouth_center[0] - eye_center[0])
                mouth_offset_ratio = mouth_offset / face_width
                
                if mouth_offset_ratio > 0.20:  # 嘴部偏移不能超过20%
                    logger.debug(f"嘴部偏移过大: {mouth_offset_ratio:.3f}")
                    return False
            
            # 4. 眼部水平对齐检查
            eye_level_diff = abs(left_eye[1] - right_eye[1])
            eye_level_ratio = eye_level_diff / face_height
            
            if eye_level_ratio > 0.1:  # 眼部高度差不能超过面部高度的10%
                logger.debug(f"眼部高度差异过大: {eye_level_ratio:.3f}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"正脸特征验证失败: {e}")
            return False
    
    def process_single_image(self, image_path: str) -> Dict:
        """处理单张图片 - 内存优化版"""
        img = None  # 提前声明图片变量
        try:
            filename = os.path.basename(image_path)
            start_time = time.time()
            
            total_score = 0
            frontal_face_count = 0
            clear_plate_count = 0
            text_count = 0
            
            # 1. RetinaFace检测正脸
            try:
                detections = RetinaFace.detect_faces(image_path)
                
                if isinstance(detections, dict) and len(detections) > 0:
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        img_area = img_width * img_height
                        
                        # 如果图像太大，缩放以节省内存
                        if img_height > 1024 or img_width > 1024:
                            scale_factor = min(1024/img_height, 1024/img_width)
                            new_height, new_width = int(img_height * scale_factor), int(img_width * scale_factor)
                            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                            # 更新图像尺寸和面积
                            img_height, img_width = img.shape[:2]
                            img_area = img_width * img_height
                        
                        for face_key, face_data in detections.items():
                            confidence = face_data.get('score', 0.0)
                            if confidence < self.config.MIN_FACE_CONFIDENCE_RETINA:
                                continue
                            
                            facial_area = face_data['facial_area']
                            landmarks = face_data.get('landmarks', {})
                            
                            if not landmarks:
                                continue
                            
                            x1, y1, x2, y2 = facial_area
                            face_width = x2 - x1
                            face_height = y2 - y1
                            face_area = face_width * face_height
                            
                            # 基础尺寸过滤 - 更严格
                            if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                                continue
                            
                            if face_area < self.config.MIN_FACE_AREA:
                                continue
                            
                            # 分辨率质量检查 - 新增
                            face_resolution = max(face_width, face_height)
                            if face_resolution < self.config.MIN_FACE_RESOLUTION:
                                continue
                            
                            # 距离图片边缘检查 - 避免边缘的远景人脸
                            face_center_x = (x1 + x2) / 2
                            face_center_y = (y1 + y2) / 2
                            
                            # 计算人脸中心到图片边缘的最小距离比例
                            edge_dist_x = min(face_center_x / img_width, (img_width - face_center_x) / img_width)
                            edge_dist_y = min(face_center_y / img_height, (img_height - face_center_y) / img_height)
                            min_edge_distance = min(edge_dist_x, edge_dist_y)
                            
                            # 如果人脸太接近边缘，可能是远景，过滤掉
                            if min_edge_distance < (1 - self.config.MAX_DISTANCE_THRESHOLD):
                                continue
                            
                            # 面积比例检查 - 更严格
                            area_ratio = face_area / img_area
                            is_close_up = area_ratio >= self.config.CLOSE_UP_FACE_RATIO
                            
                            # 额外的近景验证：人脸宽度占图片宽度的比例
                            width_ratio = face_width / img_width
                            height_ratio = face_height / img_height
                            size_ratio = max(width_ratio, height_ratio)
                            
                            # 只有足够大的人脸才被认为是近景
                            is_large_enough = size_ratio >= 0.15  # 人脸至少占图片尺寸的15%
                            
                            yaw_angle = self.calculate_yaw_angle(landmarks)
                            is_frontal = yaw_angle <= self.config.YAW_ANGLE_THRESHOLD
                            
                            # 新增：验证是否为真正的正脸特征（非后脑勺）
                            is_valid_frontal = self.validate_frontal_face_features(landmarks, facial_area)
                            
                            # 综合判断：正面 + 近景 + 足够大 + 不在边缘 + 特征验证
                            if is_frontal and is_close_up and is_large_enough and is_valid_frontal:
                                frontal_face_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_FRONTAL_FACE
                                logger.debug(f"检测到正脸: yaw={yaw_angle:.1f}°, 面积比={area_ratio:.3f}, 尺寸比={size_ratio:.3f}")
                            elif not is_valid_frontal:
                                logger.debug(f"拒绝后脑勺/侧脸: yaw={yaw_angle:.1f}°, 特征验证失败")
                            else:
                                logger.debug(f"拒绝人脸: yaw={yaw_angle:.1f}°, 面积比={area_ratio:.3f}, 尺寸比={size_ratio:.3f}")
                    
                    # 释放图像内存
                    if img is not None:
                        del img
                        img = None
                            
            except Exception as e:
                logger.debug(f"RetinaFace检测失败 {image_path}: {e}")
                if img is not None:
                    del img
                    img = None
            
            # 2. 检测车牌
            try:
                plate_results = self.models['plate']([image_path], verbose=False, device=self.device)
                
                if plate_results and len(plate_results) > 0:
                    result = plate_results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            if confidence >= self.config.MIN_PLATE_CONFIDENCE:
                                clear_plate_count += 1
                                total_score += self.config.SCORE_PER_CLEAR_PLATE
                                
            except Exception as e:
                logger.debug(f"车牌检测失败 {image_path}: {e}")
            
            # 3. 检测文字
            try:
                if self.ocr_reader is not None:
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        # 如果图像过大，缩放以节省内存
                        height, width = img.shape[:2]
                        if height > 800 or width > 800:
                            scale_factor = min(800/height, 800/width)
                            new_height, new_width = int(height * scale_factor), int(width * scale_factor)
                            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        
                        ocr_results = self.ocr_reader.readtext(img)
                        
                        # 立即释放图像内存
                        del img
                        img = None
                        
                        if ocr_results:
                            for bbox, text, confidence in ocr_results:
                                confidence = float(confidence) if confidence is not None else 0.0
                                
                                if confidence >= self.config.MIN_TEXT_CONFIDENCE:
                                    cleaned_text = text.strip()
                                    if len(cleaned_text) >= 2:
                                        text_count += 1
                                        total_score += self.config.SCORE_PER_TEXT
                                        
            except Exception as e:
                logger.debug(f"文字检测失败 {image_path}: {e}")
                if img is not None:
                    del img
                    img = None
            
            # 4. 确定分类
            if total_score > self.config.REQUIRED_TOTAL_SCORE:
                category = 'high_score'
            elif total_score > 0:
                category = 'low_score'
            else:
                category = 'zero_score'
            
            processing_time = time.time() - start_time
            
            return {
                'filename': filename,
                'category': category,
                'total_score': total_score,
                'frontal_faces': frontal_face_count,
                'clear_plates': clear_plate_count,
                'texts': text_count,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            return {
                'filename': os.path.basename(image_path),
                'category': 'failed',
                'error': str(e),
                'total_score': 0
            }
        finally:
            # 确保图像内存被释放
            if img is not None:
                del img
            # 强制垃圾回收
            gc.collect()
    
    def process_batch(self, image_paths: List[str]) -> List[Tuple[str, Dict]]:
        """批量处理"""
        results = []
        for image_path in image_paths:
            result = self.process_single_image(image_path)
            results.append((image_path, result))
        return results
    
    def move_image(self, image_path: str, category: str, output_dirs: Dict[str, str]) -> bool:
        """移动图片到分类目录"""
        try:
            filename = os.path.basename(image_path)
            output_dir = output_dirs[category]
            output_path = os.path.join(output_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
            
            shutil.copy2(image_path, output_path)
            os.remove(image_path)
            return True
            
        except Exception as e:
            logger.error(f"移动图片失败 {image_path}: {e}")
            return False
    
    def process_high_score_similarity(self, high_score_dir: str, output_dirs: Dict[str, str]):
        """处理高分图片的相似度检测和去重"""
        if not self.config.ENABLE_SIMILARITY_DETECTION or not self.similarity_detector:
            logger.info("⚠️ 相似图片检测已禁用，跳过高分图片去重")
            return
        
        logger.info("🔍 开始处理高分图片相似度检测...")
        
        # 获取高分图片文件列表
        high_score_images = []
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            high_score_images.extend(Path(high_score_dir).glob(pattern))
        
        high_score_images = sorted([str(f) for f in high_score_images if f.is_file()])
        
        if len(high_score_images) == 0:
            logger.info("❌ 未找到高分图片，跳过相似度检测")
            return
        
        logger.info(f"📊 找到 {len(high_score_images)} 张高分图片进行相似度检测")
        
        try:
            # 执行相似度检测和自动去重
            unique_images, similar_pairs = self.similarity_detector.auto_deduplicate(
                high_score_images,
                output_dirs['unique_high_score'],
                output_dirs['similar_high_score']
            )
            
            # 更新统计信息
            self.stats['unique_high_score'] = len(unique_images)
            self.stats['similar_high_score'] = len(similar_pairs)
            
            logger.info("✅ 高分图片相似度检测完成!")
            logger.info(f"   - 去重后保留: {self.stats['unique_high_score']} 张")
            logger.info(f"   - 相似图片: {self.stats['similar_high_score']} 张")
            logger.info(f"   - 压缩率: {(1 - self.stats['unique_high_score'] / len(high_score_images)) * 100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ 高分图片相似度检测失败: {e}")
            logger.info("🔄 将原始高分图片复制到unique目录...")
            
            # 失败时将所有高分图片复制到unique目录
            for img_path in high_score_images:
                try:
                    filename = os.path.basename(img_path)
                    dst_path = os.path.join(output_dirs['unique_high_score'], filename)
                    shutil.copy2(img_path, dst_path)
                    self.stats['unique_high_score'] += 1
                except Exception as copy_e:
                    logger.error(f"复制图片失败 {img_path}: {copy_e}")
    
    def generate_similarity_report(self, output_dirs: Dict[str, str]):
        """生成相似度检测报告"""
        if not self.similarity_detector:
            return
        
        try:
            report_file = os.path.join(output_dirs['analysis'], 'similarity_detection_report.txt')
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("GPU加速相似图片检测报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("配置信息:\n")
                f.write(f"  - PSNR阈值: {self.config.PSNR_THRESHOLD} dB\n")
                f.write(f"  - 相邻帧阈值: {self.config.ADJACENT_FRAME_THRESHOLD} 帧\n")
                f.write(f"  - 最小帧间距: {self.config.MIN_FRAME_DISTANCE} 帧\n")
                f.write(f"  - GPU加速: {'启用' if self.similarity_detector.use_gpu else '禁用'}\n\n")
                
                f.write("检测统计:\n")
                f.write(f"  - 处理的图片对: {self.similarity_detector.stats['processed_pairs']}\n")
                f.write(f"  - GPU操作次数: {self.similarity_detector.stats['gpu_operations']}\n")
                f.write(f"  - CPU操作次数: {self.similarity_detector.stats['cpu_operations']}\n")
                f.write(f"  - 相邻帧相似检测: {self.similarity_detector.stats['adjacent_frames_similar']}\n")
                f.write(f"  - 内存清理次数: {self.similarity_detector.stats['memory_cleanups']}\n\n")
                
                f.write("去重结果:\n")
                f.write(f"  - 去重前高分图片: {self.stats['high_score']}\n")
                f.write(f"  - 去重后唯一图片: {self.stats['unique_high_score']}\n")
                f.write(f"  - 识别为相似的图片: {self.stats['similar_high_score']}\n")
                
                if self.stats['high_score'] > 0:
                    compression_rate = (1 - self.stats['unique_high_score'] / self.stats['high_score']) * 100
                    f.write(f"  - 压缩率: {compression_rate:.1f}%\n")
            
            logger.info(f"📄 相似度检测报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 生成相似度检测报告失败: {e}")
    
    def run(self):
        """运行快速检测器"""
        logger.info("🚀 启动快速人脸车牌检测器...")
        logger.info(f"📁 输入目录: {self.config.INPUT_DIR}")
        logger.info(f"📦 初始批处理大小: {self.config.BATCH_SIZE} (将动态调整)")
        logger.info(f"🎯 预过滤: {'启用' if self.config.ENABLE_PREFILTER else '禁用'}")
        logger.info(f"🔍 相似图片检测: {'启用' if self.config.ENABLE_SIMILARITY_DETECTION else '禁用'}")
        logger.info(f"👤 改进正脸检测: 置信度≥{self.config.MIN_FACE_CONFIDENCE_RETINA}, yaw≤{self.config.YAW_ANGLE_THRESHOLD}°")
        logger.info(f"👤 人脸过滤策略: 最小尺寸{self.config.MIN_FACE_SIZE}px, 面积比例≥{self.config.CLOSE_UP_FACE_RATIO:.1%}")
        logger.info(f"🔍 特征验证: 启用后脑勺检测和面部对称性验证")
        logger.info(f"🔍 远景过滤: 最小分辨率{self.config.MIN_FACE_RESOLUTION}px, 边缘距离≥{self.config.MAX_DISTANCE_THRESHOLD:.1%}")
        logger.info(f"💾 内存限制: GPU {self.config.MAX_GPU_MEMORY_MB}MB, CPU {self.config.MAX_CPU_MEMORY_MB}MB")
        logger.info(f"🧹 内存清理: 每{self.config.MEMORY_CLEANUP_INTERVAL}个批次强制清理")
        
        if self.config.ENABLE_SIMILARITY_DETECTION:
            logger.info(f"📊 相似度参数: PSNR阈值{self.config.PSNR_THRESHOLD}dB, 相邻帧阈值{self.config.ADJACENT_FRAME_THRESHOLD}帧")
        
        # 初始化模型
        if not self.initialize_models():
            logger.error("❌ 模型初始化失败")
            return
        
        # 创建输出目录
        output_dirs = self.config.get_output_dirs()
        for name, dir_path in output_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
        
        # 获取图像文件
        input_path = Path(self.config.INPUT_DIR)
        image_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            image_files.extend(input_path.glob(pattern))
        
        image_files = sorted([str(f) for f in image_files if f.is_file()])
        logger.info(f"📊 找到 {len(image_files)} 张图片")
        
        if not image_files:
            logger.warning("❌ 未找到图片文件")
            return
        
        # 批量处理 - 内存优化版
        progress = SimpleProgressBar(len(image_files), "处理进度")
        processed = 0
        memory_manager = GPUMemoryManager()  # 创建内存管理器
        
        for i in range(0, len(image_files), self.config.BATCH_SIZE):
            # 动态调整批处理大小
            current_batch_size = memory_manager.get_dynamic_batch_size(self.config.BATCH_SIZE)
            batch = image_files[i:i + current_batch_size]
            
            try:
                batch_results = self.process_batch(batch)
                
                for image_path, result in batch_results:
                    category = result.get('category', 'failed')
                    if category != 'failed':
                        if self.move_image(image_path, category, output_dirs):
                            self.stats[category] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    processed += 1
                
                progress.update(processed)
                
                # 更频繁的内存清理
                batch_number = (i // current_batch_size) + 1
                if batch_number % self.config.MEMORY_CLEANUP_INTERVAL == 0:
                    logger.info("🧹 执行内存清理...")
                    if self.device and 'cuda' in self.device:
                        torch.cuda.empty_cache()
                    memory_manager.force_gpu_cleanup()
                    # 强制Python垃圾回收
                    for _ in range(3):
                        gc.collect()
                    logger.info("✅ 内存清理完成")
                    
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                for image_path in batch:
                    self.stats['failed'] += 1
                    processed += 1
                progress.update(processed)
        
        # 打印统计结果
        logger.info("="*60)
        logger.info("🎉 基础检测完成！统计结果:")
        logger.info(f"✅ 高分图片(>5分): {self.stats['high_score']:,} 张")
        logger.info(f"📊 低分图片(1-5分): {self.stats['low_score']:,} 张")
        logger.info(f"❌ 零分图片(0分): {self.stats['zero_score']:,} 张")
        logger.info(f"❌ 处理失败: {self.stats['failed']:,} 张")
        
        total = sum([self.stats['high_score'], self.stats['low_score'], self.stats['zero_score'], self.stats['failed']])
        if total > 0:
            success_rate = (self.stats['high_score'] / total) * 100
            logger.info(f"📈 符合要求比例: {success_rate:.1f}%")
        
        # 处理高分图片的相似度检测
        if self.config.ENABLE_SIMILARITY_DETECTION and self.stats['high_score'] > 0:
            logger.info("="*60)
            logger.info("🔍 开始高分图片相似度检测和去重...")
            
            self.process_high_score_similarity(
                output_dirs['high_score'], 
                output_dirs
            )
            
            # 生成相似度检测报告
            self.generate_similarity_report(output_dirs)
            
            logger.info("="*60)
            logger.info("🎉 相似度检测完成！最终统计:")
            logger.info(f"✅ 原始高分图片: {self.stats['high_score']:,} 张")
            logger.info(f"🎯 去重后唯一图片: {self.stats['unique_high_score']:,} 张")
            logger.info(f"📋 相似图片: {self.stats['similar_high_score']:,} 张")
            
            if self.stats['high_score'] > 0:
                final_compression = (1 - self.stats['unique_high_score'] / self.stats['high_score']) * 100
                logger.info(f"📈 最终压缩率: {final_compression:.1f}%")
        else:
            logger.info("⚠️ 相似图片检测已禁用或无高分图片")
        
        logger.info("="*60)

def get_image_files(input_dir):
    """获取目录中的所有图片文件并检查完整性"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    image_files = []
    corrupted_files = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"❌ 输入目录不存在: {input_dir}")
        return []
    
    logger.info(f"🔍 扫描图片文件...")
    all_files = []
    for ext in image_extensions:
        all_files.extend(input_path.glob(f'*{ext}'))
        all_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    logger.info(f"📁 找到 {len(all_files)} 个图片文件，检查完整性...")
    
    for file_path in all_files:
        try:
            # 简单的图片完整性检查
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                image_files.append(file_path)
            else:
                corrupted_files.append(file_path)
        except Exception as e:
            logger.debug(f"图片读取失败 {file_path}: {e}")
            corrupted_files.append(file_path)
    
    if corrupted_files:
        logger.warning(f"❌ 发现 {len(corrupted_files)} 个损坏或无法读取的文件:")
        for corrupted_file in corrupted_files[:10]:
            logger.warning(f"   - {corrupted_file.name}")
        if len(corrupted_files) > 10:
            logger.warning(f"   - ... 还有 {len(corrupted_files) - 10} 个文件")
        logger.info(f"✅ 有效图片文件: {len(image_files)} 个")
    else:
        logger.info(f"✅ 所有 {len(image_files)} 个图片文件都有效")
    
    return sorted(image_files)


def get_user_input_directory():
    """获取用户输入的目录路径"""
    print("🚀 快速人脸车牌检测器 - 交互式版本")
    print("=" * 60)
    
    while True:
        print("\n📁 请输入要处理的图片目录路径:")
        print("   (输入 'q' 或 'quit' 退出程序)")
        print("   (输入 'help' 查看帮助信息)")
        
        user_input = input("👉 图片目录路径: ").strip()
        
        # 处理退出命令
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("👋 程序退出")
            return None
        
        # 处理帮助命令
        if user_input.lower() == 'help':
            print("\n💡 使用说明:")
            print("   - 请输入包含图片的完整目录路径")
            print("   - 支持的图片格式: .jpg, .jpeg, .png, .bmp, .webp")
            print("   - 输出将保存在输入目录下的 'processed_output' 文件夹中")
            print("   - 程序会自动检测人脸、车牌和文字，并进行相似图片去重")
            print("   - 示例路径: /home/zhiqics/sanjian/predata/output_frames70")
            continue
        
        # 验证输入
        if not user_input:
            print("❌ 请输入有效的目录路径")
            continue
        
        # 展开用户目录符号
        if user_input.startswith('~'):
            user_input = os.path.expanduser(user_input)
        
        # 转换为绝对路径
        abs_path = os.path.abspath(user_input)
        
        # 检查目录是否存在
        if not os.path.exists(abs_path):
            print(f"❌ 目录不存在: {abs_path}")
            print("   请检查路径是否正确")
            continue
        
        if not os.path.isdir(abs_path):
            print(f"❌ 路径不是目录: {abs_path}")
            continue
        
        # 检查是否有读取权限
        if not os.access(abs_path, os.R_OK):
            print(f"❌ 没有读取权限: {abs_path}")
            continue
        
        # 预览目录内容
        try:
            image_files = []
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            
            for file in os.listdir(abs_path):
                file_path = os.path.join(abs_path, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in supported_formats:
                        image_files.append(file)
            
            if len(image_files) == 0:
                print(f"⚠️ 目录中没有找到支持的图片文件: {abs_path}")
                print(f"   支持的格式: {', '.join(supported_formats)}")
                
                choice = input("是否继续处理此目录? (y/n): ").strip().lower()
                if choice not in ['y', 'yes']:
                    continue
            else:
                print(f"✅ 找到 {len(image_files)} 张图片文件")
                print(f"   目录: {abs_path}")
                
                # 显示前几个文件名作为预览
                preview_count = min(5, len(image_files))
                print(f"   预览前{preview_count}个文件:")
                for i, filename in enumerate(image_files[:preview_count]):
                    print(f"     {i+1}. {filename}")
                
                if len(image_files) > preview_count:
                    print(f"     ... 还有 {len(image_files) - preview_count} 个文件")
        
        except Exception as e:
            print(f"❌ 读取目录失败: {e}")
            continue
        
        # 显示输出目录
        output_dir = os.path.join(abs_path, "processed_output")
        print(f"\n📤 输出目录: {output_dir}")
        
        # 确认处理
        print(f"\n🔍 配置预览:")
        print(f"   - 输入目录: {abs_path}")
        print(f"   - 输出目录: {output_dir}")
        print(f"   - 图片数量: {len(image_files) if 'image_files' in locals() else '未知'}")
        print(f"   - 相似图片检测: 启用")
        print(f"   - GPU加速: {'启用' if torch.cuda.is_available() else '禁用'}")
        
        confirm = input("\n确认开始处理? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return abs_path
        else:
            print("取消处理，请重新输入目录路径")


def main():
    """主函数"""
    try:
        # 获取用户输入的目录
        input_dir = get_user_input_directory()
        
        if input_dir is None:
            return  # 用户选择退出
        
        # 创建配置
        config = FastConfig(input_dir)
        
        print(f"\n🔧 初始化检测器...")
        print(f"📁 输入目录: {config.INPUT_DIR}")
        print(f"📁 输出目录: {config.OUTPUT_BASE_DIR}")
        
        # 检查模型文件
        missing_models = []
        if not os.path.exists(config.YOLOV8S_MODEL_PATH):
            missing_models.append(f"YOLOv8s模型: {config.YOLOV8S_MODEL_PATH}")
        
        if not os.path.exists(config.LICENSE_PLATE_MODEL_PATH):
            missing_models.append(f"车牌检测模型: {config.LICENSE_PLATE_MODEL_PATH}")
        
        if missing_models:
            print(f"\n⚠️ 以下模型文件不存在:")
            for model in missing_models:
                print(f"   - {model}")
            
            print(f"\n💡 建议操作:")
            print(f"   1. 检查模型文件路径是否正确")
            print(f"   2. 下载缺失的模型文件")
            print(f"   3. 或者程序将尝试自动下载默认模型")
            
            choice = input("\n是否继续运行? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("程序退出")
                return
        
        # 运行检测器
        processor = FastProcessor(config)
        processor.run()
        
        print(f"\n🎉 处理完成!")
        print(f"📁 结果保存在: {config.OUTPUT_BASE_DIR}")
        print(f"📊 请查看各分类目录中的图片和分析报告")
        
    except KeyboardInterrupt:
        logger.info("⚡ 用户中断操作")
        print("\n⚡ 操作被用户中断")
    except Exception as e:
        logger.error(f"❌ 程序执行错误: {e}")
        print(f"\n❌ 程序执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("详细错误信息已保存到日志文件")


if __name__ == "__main__":
    main()
