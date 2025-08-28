#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速的相似图片检测工具
基于PSNR算法检测并处理相似图片，使用CUDA加速
支持自动去重、手动选择、相似图片分组等功能
"""

import os
import cv2
import shutil
import numpy as np
import math
import argparse
import gc
import psutil
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy可用，将使用GPU加速")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy不可用，将使用CPU处理")

class GPUMemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, max_gpu_memory_mb=4096, max_cpu_memory_mb=2048, warning_threshold=0.8):
        """
        初始化GPU内存管理器
        
        Args:
            max_gpu_memory_mb (int): 最大GPU内存使用量(MB)
            max_cpu_memory_mb (int): 最大CPU内存使用量(MB)
            warning_threshold (float): 内存警告阈值(0-1)
        """
        self.max_gpu_memory_bytes = max_gpu_memory_mb * 1024 * 1024
        self.max_cpu_memory_bytes = max_cpu_memory_mb * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_available = self._check_gpu_availability()
        
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
        if self.gpu_available:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except:
                pass
        gc.collect()
    
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
        """内存监控线程"""
        while self.monitoring:
            cpu_memory = self.get_cpu_memory_usage()
            gpu_memory = self.get_gpu_memory_usage()
            
            # CPU内存监控
            cpu_usage_ratio = cpu_memory['rss'] / self.max_cpu_memory_bytes
            if cpu_usage_ratio > self.warning_threshold:
                print(f"⚠️ CPU内存警告: {cpu_usage_ratio:.1%} ({cpu_memory['rss']/1024/1024:.1f}MB)")
                
                if cpu_usage_ratio > 0.95:
                    print("🔴 CPU内存严重不足，执行强制清理...")
                    self.force_gpu_cleanup()
            
            # GPU内存监控
            if self.gpu_available and gpu_memory['total'] > 0:
                gpu_usage_ratio = gpu_memory['used'] / gpu_memory['total']
                if gpu_usage_ratio > self.warning_threshold:
                    print(f"⚠️ GPU内存警告: {gpu_usage_ratio:.1%} ({gpu_memory['used']/1024/1024:.1f}MB)")
                    
                    if gpu_usage_ratio > 0.95:
                        print("🔴 GPU内存严重不足，执行强制清理...")
                        self.force_gpu_cleanup()
            
            time.sleep(3)  # 每3秒检查一次

class SimilarImageDetectorGPU:
    """GPU加速的相似图片检测器"""
    
    def __init__(self, psnr_threshold=50.0, max_gpu_memory_mb=4096, max_cpu_memory_mb=2048, 
                 use_gpu=True, batch_size=None, min_frame_distance=5, adjacent_frame_threshold=8):
        """
        初始化GPU加速的相似图片检测器
        
        Args:
            psnr_threshold (float): PSNR阈值
            max_gpu_memory_mb (int): 最大GPU内存使用量(MB)
            max_cpu_memory_mb (int): 最大CPU内存使用量(MB)
            use_gpu (bool): 是否使用GPU加速
            batch_size (int): 批处理大小，保留兼容性
            min_frame_distance (int): 最小帧间距，用于过滤相邻帧
            adjacent_frame_threshold (int): 相邻帧相似阈值，≤此值认定为相似
        """
        self.psnr_threshold = psnr_threshold
        self.min_frame_distance = min_frame_distance  # 新增：最小帧间距
        self.adjacent_frame_threshold = adjacent_frame_threshold  # 新增：8帧内相似阈值
        self.memory_manager = GPUMemoryManager(max_gpu_memory_mb, max_cpu_memory_mb)
        self.use_gpu = use_gpu and self.memory_manager.gpu_available
        self.batch_size = batch_size
        
        # 初始化CUDA上下文（如果使用GPU）
        if self.use_gpu:
            try:
                cp.cuda.Device(0).use()
                print(f"🚀 GPU加速已启用 (设备: {cp.cuda.Device(0).name})")
            except Exception as e:
                print(f"⚠️ GPU初始化失败，回退到CPU: {str(e)}")
                self.use_gpu = False
        
        if not self.use_gpu:
            print("💻 使用CPU处理模式")
        
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
            'adjacent_frames_filtered': 0,  # 原有：相邻帧过滤统计
            'adjacent_frames_similar': 0   # 新增：8帧内相邻帧相似统计
        }
    
    def extract_frame_number(self, filename):
        """
        从文件名中提取帧号
        
        Args:
            filename (str): 文件名
            
        Returns:
            int: 帧号，如果无法提取则返回-1
        """
        import re
        
        # 多种帧号匹配模式
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
    
    def are_adjacent_frames(self, path1, path2, max_distance=None):
        """
        检查两个图片是否为相邻帧
        
        Args:
            path1, path2: 图片路径
            max_distance: 最大帧间距，如果为None则使用self.min_frame_distance
            
        Returns:
            bool: 是否为相邻帧
        """
        if max_distance is None:
            max_distance = self.min_frame_distance
            
        frame1 = self.extract_frame_number(path1.name)
        frame2 = self.extract_frame_number(path2.name)
        
        # 如果无法提取帧号，则不认为是相邻帧
        if frame1 == -1 or frame2 == -1:
            return False
        
        frame_distance = abs(frame1 - frame2)
        return frame_distance < max_distance
    
    def calculate_image_clarity(self, image_path):
        """
        计算图片的清晰度分数
        
        Args:
            image_path: 图片路径
            
        Returns:
            float: 清晰度分数，越高越清晰
        """
        try:
            # 检查文件是否存在
            if not Path(image_path).exists():
                print(f"⚠️ 文件不存在: {image_path}")
                return 0.0
            
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ 无法读取图片: {image_path.name}")
                return 0.0
            
            # 检查图片尺寸
            if img.shape[0] == 0 or img.shape[1] == 0:
                print(f"⚠️ 图片尺寸无效: {image_path.name}")
                return 0.0
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU加速的清晰度计算
                try:
                    gpu_gray = cp.asarray(gray, dtype=cp.float32)
                    
                    # 使用Laplacian算子计算清晰度
                    kernel = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
                    kernel = kernel.reshape(3, 3)
                    
                    # 卷积计算
                    from cupyx.scipy import ndimage
                    laplacian = ndimage.convolve(gpu_gray, kernel)
                    clarity = float(cp.var(laplacian))
                    
                    del gpu_gray, laplacian
                    self.stats['gpu_operations'] += 1
                except Exception as gpu_error:
                    print(f"⚠️ GPU清晰度计算失败，回退到CPU: {str(gpu_error)}")
                    # 回退到CPU计算
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    clarity = laplacian.var()
                    self.stats['cpu_operations'] += 1
            else:
                # CPU计算清晰度
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                clarity = laplacian.var()
                self.stats['cpu_operations'] += 1
            
            # 结合文件大小作为质量指标
            try:
                file_size = os.path.getsize(image_path)
                quality_score = clarity * 0.8 + (file_size / 100000) * 0.2
            except OSError:
                quality_score = clarity
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️ 清晰度计算失败 {image_path.name}: {str(e)}")
            return 0.0
    
    def are_adjacent_frames_similar(self, path1, path2, frame_threshold=8):
        """
        检查两个图片是否为相邻帧且应认定为相似
        
        Args:
            path1, path2: 图片路径
            frame_threshold: 帧间距阈值，小于等于此值认定为相似
            
        Returns:
            tuple: (是否为相邻帧相似, 保留哪张图片的路径, 详细信息)
        """
        frame1 = self.extract_frame_number(path1.name)
        frame2 = self.extract_frame_number(path2.name)
        
        # 如果无法提取帧号，则不认为是相邻帧
        if frame1 == -1 or frame2 == -1:
            return False, None, {'reason': 'no_frame_number'}
        
        frame_distance = abs(frame1 - frame2)
        
        # 如果帧间距大于阈值，不认定为相似
        if frame_distance > frame_threshold:
            return False, None, {'frame_distance': frame_distance, 'threshold': frame_threshold}
        
        # 帧间距在阈值内，计算清晰度并选择保留哪张
        clarity1 = self.calculate_image_clarity(path1)
        clarity2 = self.calculate_image_clarity(path2)
        
        # 选择更清晰的图片
        if clarity1 >= clarity2:
            keep_image = path1
            remove_image = path2
        else:
            keep_image = path2
            remove_image = path1
        
        details = {
            'frame1': frame1,
            'frame2': frame2,
            'frame_distance': frame_distance,
            'threshold': frame_threshold,
            'clarity1': clarity1,
            'clarity2': clarity2,
            'keep_image': keep_image.name,
            'remove_image': remove_image.name,
            'reason': 'adjacent_frames_similar'
        }
        
        return True, keep_image, details
    
    def preprocess_image_gpu(self, img, target_size=None):
        """
        GPU加速的图片预处理
        
        Args:
            img: OpenCV图片数组
            target_size: 目标尺寸 (width, height)
            
        Returns:
            preprocessed image: 预处理后的图片
        """
        if img is None:
            return None
            
        try:
            if self.use_gpu and CUPY_AVAILABLE:
                # 使用GPU处理
                gpu_img = cp.asarray(img)
                
                if target_size is not None:
                    # GPU版本的resize（使用cupy-opencv或自定义插值）
                    # 注意：cupy没有直接的resize函数，这里使用CPU版本
                    cpu_img = cp.asnumpy(gpu_img)
                    resized = cv2.resize(cpu_img, target_size)
                    result = cp.asarray(resized)
                else:
                    result = gpu_img.copy()
                
                self.stats['gpu_operations'] += 1
                if CUPY_AVAILABLE:
                    return cp.asnumpy(result)  # 返回CPU数组用于后续处理
                else:
                    return result
            else:
                # CPU处理
                if target_size is not None:
                    result = cv2.resize(img, target_size)
                else:
                    result = img.copy()
                
                self.stats['cpu_operations'] += 1
                return result
                
        except Exception as e:
            print(f"⚠️ 图片预处理失败，使用CPU: {str(e)}")
            # 回退到CPU处理
            if target_size is not None:
                return cv2.resize(img, target_size)
            else:
                return img.copy()
    
    def calculate_psnr_gpu(self, img1, img2):
        """
        GPU加速的PSNR计算
        
        Args:
            img1, img2: OpenCV图片数组
            
        Returns:
            float: PSNR值
        """
        try:
            # 处理不同尺寸的图片
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
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU加速的MSE计算
                gpu_img1 = cp.asarray(img1_cropped, dtype=cp.float32)
                gpu_img2 = cp.asarray(img2_cropped, dtype=cp.float32)
                
                # 计算MSE
                diff = gpu_img1 - gpu_img2
                mse = cp.mean(diff * diff)
                mse_cpu = cp.asnumpy(mse).item()
                
                self.stats['gpu_operations'] += 1
            else:
                # CPU计算
                mse_cpu = np.mean((img1_cropped.astype(np.float32) - img2_cropped.astype(np.float32)) ** 2)
                self.stats['cpu_operations'] += 1
            
            if mse_cpu == 0:
                return float('inf')
            
            # 计算PSNR
            max_pixel_value = 255.0
            psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse_cpu))
            
            return psnr
            
        except Exception as e:
            print(f"⚠️ GPU PSNR计算失败，使用CPU: {str(e)}")
            # 回退到CPU计算
            return self.calculate_psnr_cpu_fallback(img1, img2)
        finally:
            # 清理GPU内存
            if self.use_gpu:
                try:
                    if 'gpu_img1' in locals():
                        del gpu_img1
                    if 'gpu_img2' in locals():
                        del gpu_img2
                    if 'diff' in locals():
                        del diff
                except:
                    pass
            
            # 清理CPU变量
            try:
                if 'img1_cropped' in locals():
                    del img1_cropped
                if 'img2_cropped' in locals():
                    del img2_cropped
            except:
                pass
    
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
                return float('inf')
            
            max_pixel_value = 255.0
            psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
            
            self.stats['cpu_operations'] += 1
            return psnr
            
        except Exception as e:
            print(f"⚠️ CPU PSNR计算出错: {str(e)}")
            return 0.0
    
    def calculate_ssim_gpu(self, img1, img2):
        """
        GPU加速的SSIM计算
        
        Args:
            img1, img2: OpenCV图片数组
            
        Returns:
            float: SSIM值
        """
        try:
            if img1 is None or img2 is None:
                return 0.0
            
            # 处理不同尺寸
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
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2GRAY)
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU加速的SSIM计算
                gpu_gray1 = cp.asarray(gray1, dtype=cp.float32)
                gpu_gray2 = cp.asarray(gray2, dtype=cp.float32)
                
                # 使用GPU计算均值和方差
                mu1 = cp.mean(gpu_gray1)
                mu2 = cp.mean(gpu_gray2)
                
                sigma1_sq = cp.var(gpu_gray1)
                sigma2_sq = cp.var(gpu_gray2)
                
                # 协方差计算
                sigma12 = cp.mean((gpu_gray1 - mu1) * (gpu_gray2 - mu2))
                
                # SSIM常数
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2
                
                # SSIM公式
                ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
                
                result = float(cp.asnumpy(ssim_value))
                self.stats['gpu_operations'] += 1
                
            else:
                # CPU计算（简化版）
                mu1 = np.mean(gray1)
                mu2 = np.mean(gray2)
                
                sigma1_sq = np.var(gray1)
                sigma2_sq = np.var(gray2)
                sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
                
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2
                
                result = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                        ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
                
                self.stats['cpu_operations'] += 1
            
            return result
            
        except Exception as e:
            print(f"⚠️ SSIM计算出错: {str(e)}")
            return 0.0
        finally:
            # 清理GPU内存
            if self.use_gpu:
                try:
                    if 'gpu_gray1' in locals():
                        del gpu_gray1
                    if 'gpu_gray2' in locals():
                        del gpu_gray2
                except:
                    pass
    
    def calculate_histogram_similarity_gpu(self, img1, img2):
        """
        GPU加速的直方图相似度计算
        
        Args:
            img1, img2: OpenCV图片数组
            
        Returns:
            float: 相关性系数
        """
        try:
            if img1 is None or img2 is None:
                return 0.0
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU加速的直方图计算
                gpu_img1 = cp.asarray(img1)
                gpu_img2 = cp.asarray(img2)
                
                # 计算各通道直方图
                hist1_b = cp.histogram(gpu_img1[:,:,0], bins=256, range=(0, 256))[0]
                hist1_g = cp.histogram(gpu_img1[:,:,1], bins=256, range=(0, 256))[0]
                hist1_r = cp.histogram(gpu_img1[:,:,2], bins=256, range=(0, 256))[0]
                
                hist2_b = cp.histogram(gpu_img2[:,:,0], bins=256, range=(0, 256))[0]
                hist2_g = cp.histogram(gpu_img2[:,:,1], bins=256, range=(0, 256))[0]
                hist2_r = cp.histogram(gpu_img2[:,:,2], bins=256, range=(0, 256))[0]
                
                # 计算相关性
                corr_b = cp.corrcoef(hist1_b, hist2_b)[0, 1]
                corr_g = cp.corrcoef(hist1_g, hist2_g)[0, 1]
                corr_r = cp.corrcoef(hist1_r, hist2_r)[0, 1]
                
                # 处理NaN值
                corr_b = 0.0 if cp.isnan(corr_b) else float(cp.asnumpy(corr_b))
                corr_g = 0.0 if cp.isnan(corr_g) else float(cp.asnumpy(corr_g))
                corr_r = 0.0 if cp.isnan(corr_r) else float(cp.asnumpy(corr_r))
                
                avg_corr = (corr_b + corr_g + corr_r) / 3.0
                self.stats['gpu_operations'] += 1
                
            else:
                # CPU计算
                hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
                hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
                hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
                
                hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
                hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
                hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
                
                corr_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
                corr_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
                corr_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
                
                avg_corr = (corr_b + corr_g + corr_r) / 3.0
                self.stats['cpu_operations'] += 1
            
            return float(avg_corr)
            
        except Exception as e:
            print(f"⚠️ 直方图相似度计算出错: {str(e)}")
            return 0.0
        finally:
            # 清理GPU内存
            if self.use_gpu:
                try:
                    if 'gpu_img1' in locals():
                        del gpu_img1
                    if 'gpu_img2' in locals():
                        del gpu_img2
                except:
                    pass
    
    def are_images_similar(self, image_path1, image_path2, check_adjacent_frames=True, adjacent_frame_threshold=8):
        """
        判断两张图片是否相似（GPU加速版）
        
        Args:
            image_path1, image_path2: 图片文件路径
            check_adjacent_frames: 是否检查相邻帧并过滤
            adjacent_frame_threshold: 相邻帧阈值，小于等于此值直接认定为相似
            
        Returns:
            tuple: (是否相似, 综合相似度分数, 详细信息字典)
        """
        img1 = None
        img2 = None
        try:
            # 优先检查相邻帧：如果帧间距≤8，直接认定为相似
            if check_adjacent_frames:
                is_adjacent, keep_image, adjacent_details = self.are_adjacent_frames_similar(
                    image_path1, image_path2, adjacent_frame_threshold
                )
                
                if is_adjacent:
                    self.stats['adjacent_frames_similar'] += 1  # 更新8帧内相似统计
                    # 返回相似，但需要标记哪张图片质量更好
                    score = 100.0  # 相邻帧认定为100%相似
                    if adjacent_details is None:
                        adjacent_details = {}
                    adjacent_details['similarity_type'] = 'adjacent_frames'
                    if keep_image is not None:
                        adjacent_details['recommended_keep'] = str(keep_image.name)
                    return True, score, adjacent_details
                
                # 如果不是相邻帧，继续检查其他相邻帧过滤逻辑
                if self.are_adjacent_frames(image_path1, image_path2, self.min_frame_distance):
                    self.stats['adjacent_frames_filtered'] += 1
                    return False, 0.0, {
                        'reason': 'adjacent_frames_filtered_by_distance',
                        'frame1': self.extract_frame_number(image_path1.name),
                        'frame2': self.extract_frame_number(image_path2.name),
                        'frame_distance': abs(self.extract_frame_number(image_path1.name) - 
                                            self.extract_frame_number(image_path2.name)),
                        'min_distance_required': self.min_frame_distance
                    }
            
            # 检查内存
            if not self.memory_manager.is_gpu_memory_available():
                self.memory_manager.force_gpu_cleanup()
                self.stats['memory_cleanups'] += 1
            
            # 读取图片
            try:
                img1 = cv2.imread(str(image_path1))
                if img1 is None:
                    print(f"⚠️ 无法读取图片1: {image_path1.name}")
                    return False, 0.0, {'error': f'Failed to load image1: {image_path1.name}'}
            except Exception as e:
                print(f"⚠️ 读取图片1异常: {image_path1.name} - {str(e)}")
                return False, 0.0, {'error': f'Exception loading image1: {str(e)}'}
            
            try:
                img2 = cv2.imread(str(image_path2))
                if img2 is None:
                    print(f"⚠️ 无法读取图片2: {image_path2.name}")
                    return False, 0.0, {'error': f'Failed to load image2: {image_path2.name}'}
            except Exception as e:
                print(f"⚠️ 读取图片2异常: {image_path2.name} - {str(e)}")
                return False, 0.0, {'error': f'Exception loading image2: {str(e)}'}
            
            # 检查图片尺寸
            if img1.shape == (0, 0, 3) or img2.shape == (0, 0, 3):
                return False, 0.0, {'error': 'Invalid image dimensions'}
            
            # GPU加速的相似度计算
            psnr = self.calculate_psnr_gpu(img1, img2)
            ssim = self.calculate_ssim_gpu(img1, img2)
            hist_sim = self.calculate_histogram_similarity_gpu(img1, img2)
            
            # 综合评分
            psnr_normalized = min(psnr / 50.0, 1.0)
            composite_score = (psnr_normalized * 0.4 + ssim * 0.4 + hist_sim * 0.2) * 100
            
            # 相似性判断
            is_similar_psnr = psnr > self.psnr_threshold
            is_similar_composite = composite_score > 78.0
            is_similar_ssim = ssim > 0.8
            
            conditions_met = sum([is_similar_psnr, is_similar_composite, is_similar_ssim])
            # 更严格的判断：需要满足所有3个条件，或者PSNR和综合分数都很高
            is_similar = (conditions_met >= 3) or (is_similar_psnr and is_similar_composite and psnr > self.psnr_threshold + 5)
            
            details = {
                'psnr': psnr,
                'ssim': ssim,
                'hist_similarity': hist_sim,
                'composite_score': composite_score,
                'psnr_threshold': self.psnr_threshold,
                'conditions_met': f"{conditions_met}/3",
                'gpu_used': self.use_gpu
            }
            
            return is_similar, composite_score, details
            
        except Exception as e:
            print(f"⚠️ GPU相似度检测出错: {str(e)}")
            return False, 0.0, {'error': str(e)}
        finally:
            # 及时释放内存
            if img1 is not None:
                del img1
            if img2 is not None:
                del img2
            
            # GPU内存清理
            if self.use_gpu:
                self.memory_manager.force_gpu_cleanup()
            
            gc.collect()
    
    def find_similar_groups_with_sampling(self, image_paths, progress_callback=None, 
                                        max_similar_sequence=10, quality_check=True):
        """
        GPU加速的智能采样相似图片组检测
        
        Args:
            image_paths: 图片路径列表
            progress_callback: 进度回调函数
            max_similar_sequence: 相似序列最大长度
            quality_check: 是否启用图片质量检查
            
        Returns:
            list: 相似图片分组列表
        """
        # 开始内存监控
        self.memory_manager.start_monitoring()
        
        try:
            similar_groups = []
            total_comparisons = len(image_paths) - 1
            current_comparison = 0
            
            print(f"🚀 开始GPU加速相似图片检测...")
            print(f"   - 图片总数: {len(image_paths)}")
            print(f"   - PSNR阈值: {self.psnr_threshold} dB")
            print(f"   - GPU加速: {'启用' if self.use_gpu else '禁用'}")
            print(f"   - 最大相似序列长度: {max_similar_sequence}")
            print(f"   - 质量检查: {'启用' if quality_check else '禁用'}")
            print(f"   - 8帧内相似检测: 启用 (≤8帧认定为相似)")
            print(f"   - 最小帧间距: {self.min_frame_distance} 帧")
            print("-" * 60)
            
            if len(image_paths) == 0:
                return similar_groups
            
            i = 0
            while i < len(image_paths):
                current_group = [image_paths[i]]
                reference_img = image_paths[i]
                j = i + 1
                consecutive_similar = 0
                
                print(f"🔍 检测序列，基准: {reference_img.name} (第{i+1}张)")
                
                while j < len(image_paths):
                    current_comparison += 1
                    self.stats['processed_pairs'] += 1
                    
                    compare_img = image_paths[j]
                    
                    # 进度显示
                    if current_comparison % 10 == 0:  # GPU加速，更频繁的进度更新
                        progress_pct = (current_comparison / total_comparisons) * 100
                        cpu_memory = self.memory_manager.get_cpu_memory_usage()
                        gpu_memory = self.memory_manager.get_gpu_memory_usage()
                        
                        print(f"   ⏳ 进度: {progress_pct:.1f}% "
                              f"CPU: {cpu_memory['rss']/1024/1024:.1f}MB "
                              f"GPU: {gpu_memory['used']/1024/1024:.1f}MB")
                    
                    is_similar, score, details = self.are_images_similar(reference_img, compare_img)
                    
                    if is_similar:
                        consecutive_similar += 1
                        current_group.append(compare_img)
                        
                        # 检查是否为相邻帧相似
                        similarity_type = details.get('similarity_type', 'normal')
                        if similarity_type == 'adjacent_frames':
                            recommended_keep = details.get('recommended_keep', 'unknown')
                            frame_distance = details.get('frame_distance', 'unknown')
                            print(f"     🎯 相邻帧相似 ({consecutive_similar}): {compare_img.name} "
                                  f"(帧距: {frame_distance}, 推荐保留: {recommended_keep})")
                        else:
                            gpu_indicator = "🚀" if details.get('gpu_used', False) else "💻"
                            print(f"     ✅ 相似 ({consecutive_similar}): {compare_img.name} "
                                  f"(分数: {score:.1f}%) {gpu_indicator}")
                        
                        if consecutive_similar >= max_similar_sequence:
                            print(f"     ⚠️ 检测到长相似序列 ({consecutive_similar}张)，启动智能采样...")
                            
                            sampled_group = self._sample_from_sequence(
                                current_group, 
                                max_keep=max(2, max_similar_sequence // 4),  # 更严格的采样
                                quality_check=quality_check
                            )
                            
                            print(f"     📊 采样结果: {len(current_group)} → {len(sampled_group)} 张图片")
                            current_group = sampled_group
                            break
                        
                        j += 1
                    else:
                        print(f"     ❌ 序列结束: {compare_img.name} (分数: {score:.1f}%)")
                        break
                
                # 质量筛选
                if len(current_group) > 3 and quality_check:
                    filtered_group = self._filter_by_quality(current_group, keep_ratio=0.45)
                    if len(filtered_group) < len(current_group):
                        print(f"     🎯 质量筛选: {len(current_group)} → {len(filtered_group)} 张图片")
                        current_group = filtered_group
                
                similar_groups.append(current_group)
                
                if len(current_group) > 1:
                    print(f"   📂 完成组: {len(current_group)} 张图片")
                
                i = j if j < len(image_paths) else len(image_paths)
            
            # 更新统计信息
            self.stats['total_images'] = len(image_paths)
            self.stats['similar_groups'] = len([g for g in similar_groups if len(g) > 1])
            self.stats['similar_images'] = sum(len(g) - 1 for g in similar_groups if len(g) > 1)
            self.stats['unique_images'] = sum(len(g) for g in similar_groups)
            
            print("-" * 60)
            print(f"📊 GPU加速检测完成:")
            print(f"   - 发现 {self.stats['similar_groups']} 个相似序列")
            print(f"   - 保留 {self.stats['unique_images']} 张代表图片")
            print(f"   - 过滤 {len(image_paths) - self.stats['unique_images']} 张冗余图片")
            print(f"   - 压缩率: {(1 - self.stats['unique_images'] / len(image_paths)) * 100:.1f}%")
            print(f"   - GPU操作: {self.stats['gpu_operations']}, CPU操作: {self.stats['cpu_operations']}")
            
            return similar_groups
            
        finally:
            self.memory_manager.stop_monitoring()
    
    def _sample_from_sequence(self, sequence, max_keep=3, quality_check=True):  # 默认保留更少
        """
        GPU加速的序列采样，增加帧间距检查和相邻帧优化
        """
        if len(sequence) <= max_keep:
            return sequence
        
        try:
            # 优先处理8帧内的相邻帧：直接按清晰度选择最好的
            adjacent_frame_groups = []
            processed_indices = set()
            
            for i in range(len(sequence)):
                if i in processed_indices:
                    continue
                    
                current_adjacent_group = [sequence[i]]
                processed_indices.add(i)
                
                # 查找8帧内的相邻图片
                for j in range(i + 1, len(sequence)):
                    if j in processed_indices:
                        continue
                        
                    is_adjacent, keep_image, details = self.are_adjacent_frames_similar(
                        sequence[i], sequence[j], frame_threshold=8
                    )
                    
                    if is_adjacent:
                        current_adjacent_group.append(sequence[j])
                        processed_indices.add(j)
                
                if len(current_adjacent_group) > 1:
                    # 从相邻帧组中选择最清晰的一张
                    best_clarity = -1
                    best_image = current_adjacent_group[0]
                    
                    for img_path in current_adjacent_group:
                        clarity = self.calculate_image_clarity(img_path)
                        if clarity > best_clarity:
                            best_clarity = clarity
                            best_image = img_path
                    
                    adjacent_frame_groups.append(best_image)
                    print(f"     🎯 相邻帧组优化: {len(current_adjacent_group)} → 1 (保留最清晰)")
                else:
                    adjacent_frame_groups.append(current_adjacent_group[0])
            
            # 如果经过相邻帧优化后数量已经满足要求
            if len(adjacent_frame_groups) <= max_keep:
                return adjacent_frame_groups
            
            # 如果还是太多，继续按原有逻辑进行帧间距采样
            sequence = adjacent_frame_groups
            
            # 如果启用帧间距检查，优先保留帧间距大的图片
            if self.min_frame_distance > 1:
                # 提取帧号信息
                frame_info = []
                for img_path in sequence:
                    frame_num = self.extract_frame_number(img_path.name)
                    frame_info.append((img_path, frame_num))
                
                # 按帧号排序
                frame_info.sort(key=lambda x: x[1] if x[1] != -1 else float('inf'))
                
                # 智能采样：确保保留的图片之间有足够的帧间距
                selected = [frame_info[0][0]]  # 保留第一张
                last_frame = frame_info[0][1]
                
                for img_path, frame_num in frame_info[1:]:
                    if len(selected) >= max_keep:
                        break
                    
                    # 如果帧号有效且与上一个选中的帧距离足够大
                    if frame_num != -1 and last_frame != -1:
                        if abs(frame_num - last_frame) >= self.min_frame_distance:
                            selected.append(img_path)
                            last_frame = frame_num
                    elif frame_num == -1:  # 无法提取帧号的图片也可能被选中
                        selected.append(img_path)
                
                # 如果通过帧间距筛选后数量不够，用质量补充
                if len(selected) < max_keep and quality_check:
                    remaining = [img for img in sequence if img not in selected]
                    if remaining:
                        quality_selected = self._filter_by_quality(remaining, keep_ratio=1.0)[:max_keep - len(selected)]
                        selected.extend(quality_selected)
                
                print(f"     🎯 帧间距采样: {len(sequence)} → {len(selected)} (最小间距: {self.min_frame_distance})")
                return selected
            
            # 原有的质量评估采样逻辑
            if quality_check and self.use_gpu:
                quality_scores = []
                for img_path in sequence:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            if CUPY_AVAILABLE:
                                # GPU加速的清晰度计算
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                gpu_gray = cp.asarray(gray, dtype=cp.float32)
                                
                                # GPU Laplacian算子
                                kernel = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
                                kernel = kernel.reshape(3, 3)
                                
                                # 使用卷积计算Laplacian
                                from cupyx.scipy import ndimage
                                laplacian = ndimage.convolve(gpu_gray, kernel)
                                clarity = float(cp.var(laplacian))
                                
                                del gpu_gray, laplacian
                            else:
                                # CPU回退
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                            
                            file_size = os.path.getsize(img_path)
                            quality = clarity * 0.7 + (file_size / 10000) * 0.3
                        else:
                            quality = 0
                    except Exception as e:
                        print(f"     ⚠️ 质量计算失败: {e}")
                        quality = 0
                    
                    quality_scores.append((img_path, quality))
                
                # 按质量排序并选择
                quality_scores.sort(key=lambda x: x[1], reverse=True)
                
                result = [sequence[0]]  # 保留第一张
                
                for img_path, _ in quality_scores[:max_keep-2]:
                    if img_path not in result:
                        result.append(img_path)
                        if len(result) >= max_keep - 1:
                            break
                
                if len(sequence) > 1 and sequence[-1] not in result:
                    if len(result) < max_keep:
                        result.append(sequence[-1])
                    else:
                        result[-1] = sequence[-1]
                
                return result
            else:
                # 简单采样
                step = len(sequence) / max_keep
                indices = [int(i * step) for i in range(max_keep)]
                indices = list(set(indices))
                indices.sort()
                
                return [sequence[i] for i in indices if i < len(sequence)]
                
        except Exception as e:
            print(f"   ⚠️ GPU采样失败: {str(e)}，使用简单采样")
            step = max(1, len(sequence) // max_keep)
            return sequence[::step][:max_keep]
    
    def _filter_by_quality(self, images, keep_ratio=0.4):  # 更严格的默认比例
        """GPU加速的质量筛选"""
        if len(images) <= 2:
            return images
            
        try:
            keep_count = max(1, int(len(images) * keep_ratio))  # 至少保留1张
            
            scored_images = []
            for img_path in images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        if self.use_gpu and CUPY_AVAILABLE:
                            # GPU加速质量评估
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            gpu_gray = cp.asarray(gray, dtype=cp.float32)
                            clarity = float(cp.var(cp.gradient(gpu_gray)[0]))  # 使用梯度方差
                            del gpu_gray
                        else:
                            # CPU计算
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        file_size = os.path.getsize(img_path)
                        quality = clarity * 0.8 + (file_size / 10000) * 0.2
                    else:
                        quality = 0
                except Exception as e:
                    print(f"质量评估失败: {e}")
                    quality = 0
                
                scored_images.append((img_path, quality))
            
            scored_images.sort(key=lambda x: x[1], reverse=True)
            return [img for img, _ in scored_images[:keep_count]]
            
        except Exception as e:
            print(f"GPU质量筛选失败: {e}")
            return images[:max(2, int(len(images) * keep_ratio))]
    
    # 继承其他方法（自动去重、交互式去重、报告生成等）
    def find_similar_groups(self, image_paths, progress_callback=None):
        """兼容性包装"""
        return self.find_similar_groups_with_sampling(
            image_paths, 
            progress_callback, 
            max_similar_sequence=5,  # 更严格：更早触发采样
            quality_check=True
        )
    
    def auto_deduplicate(self, image_paths, output_dir, similar_dir):
        """GPU加速的自动去重"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(similar_dir, exist_ok=True)
        
        similar_groups = self.find_similar_groups(image_paths)
        
        print(f"\n🚀 开始GPU加速自动去重...")
        
        unique_images = []
        similar_pairs = []
        
        for group_idx, group in enumerate(similar_groups, 1):
            if len(group) > 1:
                representative = group[0]
                similar_images = group[1:]
                unique_images.append(representative)
                
                print(f"📂 相似组 {group_idx}: 保留 {representative.name}")
                
                for i, similar_img in enumerate(similar_images, 1):
                    original_name = similar_img.name
                    similar_file = Path(similar_dir) / original_name
                    counter = 1
                    while similar_file.exists():
                        name_parts = original_name.rsplit('.', 1)
                        if len(name_parts) == 2:
                            similar_file = Path(similar_dir) / f"{name_parts[0]}_{counter:02d}.{name_parts[1]}"
                        else:
                            similar_file = Path(similar_dir) / f"{original_name}_{counter:02d}"
                        counter += 1
                    
                    try:
                        shutil.move(str(similar_img), str(similar_file))
                        similar_pairs.append((representative, similar_img))
                        print(f"   📤 移动相似图片: {similar_img.name} → {similar_file.name}")
                    except Exception as e:
                        print(f"   ❌ 移动失败: {similar_img.name} - {str(e)}")
            else:
                unique_images.append(group[0])
        
        # 移动唯一图片
        print(f"\n📋 移动唯一图片到输出目录...")
        successful_moves = 0
        for i, unique_img in enumerate(unique_images, 1):
            original_name = unique_img.name
            output_file = Path(output_dir) / original_name
            counter = 1
            while output_file.exists():
                name_parts = original_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    output_file = Path(output_dir) / f"{name_parts[0]}_{counter:02d}.{name_parts[1]}"
                else:
                    output_file = Path(output_dir) / f"{original_name}_{counter:02d}"
                counter += 1
            
            try:
                shutil.move(str(unique_img), str(output_file))
                successful_moves += 1
                if i % 50 == 0:  # GPU加速，更频繁显示进度
                    print(f"   ⏳ 移动进度: {i}/{len(unique_images)}")
            except Exception as e:
                print(f"   ❌ 移动失败: {unique_img.name} - {str(e)}")
        
        print(f"   ✅ 成功移动: {successful_moves}/{len(unique_images)} 张图片")
        
        # GPU内存清理
        if self.use_gpu:
            self.memory_manager.force_gpu_cleanup()
        
        return unique_images, similar_pairs
    
    def interactive_deduplicate(self, image_paths):
        """交互式去重（沿用原版，加上GPU统计信息）"""
        similar_groups = self.find_similar_groups(image_paths)
        selected_images = []
        
        print(f"\n🎯 GPU加速交互式去重模式")
        print(f"    GPU操作: {self.stats['gpu_operations']}, CPU操作: {self.stats['cpu_operations']}")
        print("=" * 60)
        
        for group_idx, group in enumerate(similar_groups, 1):
            if len(group) > 1:
                print(f"\n📂 相似组 {group_idx} (共 {len(group)} 张相似图片):")
                for i, img_path in enumerate(group, 1):
                    try:
                        img = cv2.imread(str(img_path))
                        h, w = img.shape[:2]
                        file_size = os.path.getsize(img_path) / 1024
                        print(f"   {i}. {img_path.name} ({w}x{h}, {file_size:.1f}KB)")
                    except:
                        print(f"   {i}. {img_path.name} (无法读取)")
                
                while True:
                    try:
                        choice = input(f"请选择要保留的图片 (1-{len(group)}, 0=全部保留): ").strip()
                        if choice == '0':
                            selected_images.extend(group)
                            print("   ✅ 全部保留")
                            break
                        elif choice.isdigit() and 1 <= int(choice) <= len(group):
                            selected_images.append(group[int(choice) - 1])
                            print(f"   ✅ 保留: {group[int(choice) - 1].name}")
                            break
                        else:
                            print("   ❌ 输入无效，请重新选择")
                    except (ValueError, IndexError):
                        print("   ❌ 输入无效，请重新选择")
            else:
                selected_images.append(group[0])
        
        print(f"\n📊 GPU加速交互式去重完成:")
        print(f"   - 原始图片: {len(image_paths)} 张")
        print(f"   - 选择保留: {len(selected_images)} 张")
        print(f"   - GPU操作: {self.stats['gpu_operations']}, CPU操作: {self.stats['cpu_operations']}")
        
        return selected_images
    
    def generate_report(self, output_file, image_paths, similar_groups=None):
        """生成GPU加速检测报告"""
        if similar_groups is None:
            similar_groups = self.find_similar_groups(image_paths)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("GPU加速相似图片检测报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PSNR阈值: {self.psnr_threshold} dB\n")
            f.write(f"GPU加速: {'启用' if self.use_gpu else '禁用'}\n")
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    gpu_name = cp.cuda.Device(0).name
                    f.write(f"GPU设备: {gpu_name}\n")
                except:
                    f.write(f"GPU设备: 未知\n")
            f.write(f"比较策略: 连续比较模式 + 相邻帧过滤 + 8帧内相似检测\n")
            f.write(f"最小帧间距: {self.min_frame_distance} 帧\n")
            f.write(f"相邻帧阈值: 8 帧 (≤8帧认定为相似)\n")
            f.write(f"内存限制: GPU {self.memory_manager.max_gpu_memory_bytes/1024/1024:.1f}MB, ")
            f.write(f"CPU {self.memory_manager.max_cpu_memory_bytes/1024/1024:.1f}MB\n")
            f.write(f"总图片数: {self.stats['total_images']}\n")
            f.write(f"相似图片组: {self.stats['similar_groups']}\n")
            f.write(f"相似图片数: {self.stats['similar_images']}\n")
            f.write(f"唯一图片数: {self.stats['unique_images']}\n")
            f.write(f"相邻帧过滤数: {self.stats['adjacent_frames_filtered']}\n")
            f.write(f"8帧内相似检测: {self.stats['adjacent_frames_similar']}\n")
            f.write(f"GPU操作次数: {self.stats['gpu_operations']}\n")
            f.write(f"CPU操作次数: {self.stats['cpu_operations']}\n")
            f.write(f"内存清理次数: {self.stats['memory_cleanups']}\n")
            f.write(f"最大内存使用: {self.stats['max_memory_used']/1024/1024:.1f} MB\n\n")
            
            # 性能统计
            total_ops = self.stats['gpu_operations'] + self.stats['cpu_operations']
            if total_ops > 0:
                gpu_ratio = self.stats['gpu_operations'] / total_ops * 100
                f.write(f"GPU加速比例: {gpu_ratio:.1f}%\n\n")
            
            # 详细相似组信息
            f.write("相似图片组详情:\n")
            f.write("-" * 40 + "\n")
            
            for group_idx, group in enumerate(similar_groups, 1):
                if len(group) > 1:
                    f.write(f"\n组 {group_idx} (共 {len(group)} 张):\n")
                    
                    for i, img_path in enumerate(group, 1):
                        try:
                            img = cv2.imread(str(img_path))
                            h, w = img.shape[:2] if img is not None else (0, 0)
                            file_size = os.path.getsize(img_path) / 1024
                            f.write(f"  {i}. {img_path.name} ({w}x{h}, {file_size:.1f}KB)\n")
                        except Exception as e:
                            f.write(f"  {i}. {img_path.name} (读取失败: {str(e)})\n")
                    
                    if len(group) >= 2:
                        f.write("     相似度详情:\n")
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                try:
                                    _, score, details = self.are_images_similar(group[i], group[j])
                                    f.write(f"       {group[i].name} ↔ {group[j].name}:\n")
                                    
                                    # 检查是否为相邻帧相似
                                    if details.get('similarity_type') == 'adjacent_frames':
                                        f.write(f"         类型: 相邻帧相似 (8帧内)\n")
                                        f.write(f"         帧距: {details.get('frame_distance', 'unknown')}\n")
                                        f.write(f"         推荐保留: {details.get('recommended_keep', 'unknown')}\n")
                                        f.write(f"         综合分数: {score:.1f}%\n")
                                    elif 'error' in details:
                                        f.write(f"         错误: {details['error']}\n")
                                    elif details.get('reason') == 'adjacent_frames_filtered_by_distance':
                                        f.write(f"         类型: 帧间距过滤\n")
                                        f.write(f"         帧距: {details.get('frame_distance', 'unknown')}\n")
                                        f.write(f"         要求最小距离: {details.get('min_distance_required', 'unknown')}\n")
                                    else:
                                        # 正常的相似度检测结果
                                        psnr = details.get('psnr', 0.0)
                                        ssim = details.get('ssim', 0.0)
                                        hist_sim = details.get('hist_similarity', 0.0)
                                        gpu_used = details.get('gpu_used', False)
                                        
                                        f.write(f"         PSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}\n")
                                        f.write(f"         直方图: {hist_sim:.3f}, 综合分数: {score:.1f}%\n")
                                        f.write(f"         GPU处理: {'是' if gpu_used else '否'}\n")
                                        
                                except Exception as report_error:
                                    f.write(f"       {group[i].name} ↔ {group[j].name}:\n")
                                    f.write(f"         报告生成错误: {str(report_error)}\n")
        
        print(f"📄 GPU加速检测报告已保存: {output_file}")


def get_image_files(input_dir):
    """获取目录中的所有图片文件并检查完整性"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    image_files = []
    corrupted_files = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return []
    
    print(f"🔍 扫描图片文件...")
    all_files = []
    for ext in image_extensions:
        all_files.extend(input_path.glob(f'*{ext}'))
        all_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"📁 找到 {len(all_files)} 个图片文件，检查完整性...")
    
    for file_path in all_files:
        try:
            # 检查文件是否存在且可读
            if not file_path.exists():
                corrupted_files.append(str(file_path))
                continue
                
            # 检查文件大小
            if file_path.stat().st_size == 0:
                print(f"⚠️ 空文件: {file_path.name}")
                corrupted_files.append(str(file_path))
                continue
            
            # 尝试读取图片头部，验证文件完整性
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    print(f"⚠️ 损坏文件: {file_path.name}")
                    corrupted_files.append(str(file_path))
                    continue
                else:
                    # 释放内存
                    del img
            except Exception as read_error:
                print(f"⚠️ 读取错误: {file_path.name} - {str(read_error)}")
                corrupted_files.append(str(file_path))
                continue
            
            image_files.append(file_path)
            
        except Exception as e:
            print(f"⚠️ 文件检查错误: {file_path.name} - {str(e)}")
            corrupted_files.append(str(file_path))
    
    if corrupted_files:
        print(f"❌ 发现 {len(corrupted_files)} 个损坏或无法读取的文件:")
        for corrupted_file in corrupted_files[:10]:  # 只显示前10个
            print(f"   - {Path(corrupted_file).name}")
        if len(corrupted_files) > 10:
            print(f"   - ... 还有 {len(corrupted_files) - 10} 个文件")
        print(f"✅ 有效图片文件: {len(image_files)} 个")
    else:
        print(f"✅ 所有 {len(image_files)} 个图片文件都有效")
    
    return sorted(image_files)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU加速的相似图片检测工具')
    parser.add_argument('input_dir', help='输入图片目录')
    parser.add_argument('-o', '--output', default='output_unique', help='唯一图片输出目录')
    parser.add_argument('-s', '--similar', default='output_similar', help='相似图片输出目录')
    parser.add_argument('-t', '--threshold', type=float, default=50.0, help='PSNR阈值 (默认: 50.0)')
    parser.add_argument('-m', '--mode', choices=['auto', 'interactive', 'report'], 
                       default='auto', help='运行模式 (默认: auto)')
    parser.add_argument('-r', '--report', help='生成检测报告文件路径')
    parser.add_argument('--gpu-memory', type=int, default=4096, help='最大GPU内存使用量(MB) (默认: 4096)')
    parser.add_argument('--cpu-memory', type=int, default=2048, help='最大CPU内存使用量(MB) (默认: 2048)')
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速')
    parser.add_argument('--batch-size', type=int, help='批处理大小 (保留兼容性)')
    parser.add_argument('--min-frame-distance', type=int, default=10, help='最小帧间距，用于过滤相邻帧 (默认: 10)')
    parser.add_argument('--disable-frame-filter', action='store_true', help='禁用相邻帧过滤')
    parser.add_argument('--adjacent-frame-threshold', type=int, default=8, help='相邻帧相似阈值，小于等于此值认定为相似 (默认: 8)')
    
    args = parser.parse_args()
    
    print("🚀 GPU加速相似图片检测工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"PSNR阈值: {args.threshold} dB")
    print(f"运行模式: {args.mode}")
    print(f"GPU内存限制: {args.gpu_memory} MB")
    print(f"CPU内存限制: {args.cpu_memory} MB")
    print(f"GPU加速: {'禁用' if args.no_gpu else '自动检测'}")
    print(f"最小帧间距: {args.min_frame_distance} 帧")
    print(f"相邻帧过滤: {'禁用' if args.disable_frame_filter else '启用'}")
    print(f"8帧内相似阈值: {args.adjacent_frame_threshold} 帧")
    
    # 获取图片文件
    image_files = get_image_files(args.input_dir)
    
    if not image_files:
        print("❌ 未找到图片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图片")
    
    # 初始化GPU加速检测器
    detector = SimilarImageDetectorGPU(
        psnr_threshold=args.threshold,
        max_gpu_memory_mb=args.gpu_memory,
        max_cpu_memory_mb=args.cpu_memory,
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size,
        min_frame_distance=1 if args.disable_frame_filter else args.min_frame_distance,
        adjacent_frame_threshold=args.adjacent_frame_threshold
    )
    
    # 执行检测
    start_time = time.time()
    
    if args.mode == 'auto':
        unique_images, similar_pairs = detector.auto_deduplicate(
            image_files, args.output, args.similar
        )
        
        print(f"\n✅ GPU加速自动去重完成!")
        print(f"   - 唯一图片目录: {args.output}")
        print(f"   - 相似图片目录: {args.similar}")
        
    elif args.mode == 'interactive':
        selected_images = detector.interactive_deduplicate(image_files)
        
        os.makedirs(args.output, exist_ok=True)
        for i, img_path in enumerate(selected_images, 1):
            original_name = img_path.name
            output_file = Path(args.output) / original_name
            counter = 1
            while output_file.exists():
                name_parts = original_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    output_file = Path(args.output) / f"{name_parts[0]}_{counter:02d}.{name_parts[1]}"
                else:
                    output_file = Path(args.output) / f"{original_name}_{counter:02d}"
                counter += 1
            
            shutil.move(str(img_path), str(output_file))
        
        print(f"\n✅ GPU加速交互式去重完成!")
        print(f"   - 选中图片保存至: {args.output}")
        
    elif args.mode == 'report':
        similar_groups = detector.find_similar_groups(image_files)
        
        report_file = args.report or f"gpu_similarity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        detector.generate_report(report_file, image_files, similar_groups)
    
    # 生成报告（如果指定）
    if args.report and args.mode != 'report':
        detector.generate_report(args.report, image_files)
    
    # 性能统计
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 性能统计:")
    print(f"   - 处理时间: {processing_time:.2f} 秒")
    print(f"   - GPU操作: {detector.stats['gpu_operations']}")
    print(f"   - CPU操作: {detector.stats['cpu_operations']}")
    if detector.stats['processed_pairs'] > 0:
        avg_time = processing_time / detector.stats['processed_pairs']
        print(f"   - 平均每对比较: {avg_time*1000:.2f} 毫秒")


if __name__ == "__main__":
    # 默认配置运行
    import sys
    if len(sys.argv) == 1:
        input_dir = "/home/zhiqics/sanjian/predata/output_frames15/two_plates"
        output_dir = "/home/zhiqics/sanjian/predata/output_frames15/two_plates/unique"
        similar_dir = "/home/zhiqics/sanjian/predata/output_frames15/two_plates/similar"
        psnr_threshold = 55.0
        max_gpu_memory_mb = 4096
        max_cpu_memory_mb = 2048
        
        print("🚀 GPU加速相似图片检测工具 (默认配置)")
        print("=" * 60)
        print(f"输入目录: {input_dir}")
        print(f"PSNR阈值: {psnr_threshold} dB")
        print(f"GPU内存限制: {max_gpu_memory_mb} MB")
        print(f"CPU内存限制: {max_cpu_memory_mb} MB")
        print(f"最小帧间距: 15 帧 (相邻帧过滤)")
        print(f"8帧内相似检测: 启用 (≤8帧认定为相似，保留最清晰)")
        print(f"默认输出目录: {output_dir}")
        print(f"默认相似目录: {similar_dir}")
        
        image_files = get_image_files(input_dir)
        
        if not image_files:
            print("❌ 未找到图片文件")
            print("💡 使用方法:")
            print(f"   python {sys.argv[0]} <输入目录> [选项]")
            print("   或者修改代码中的默认目录路径")
            sys.exit(1)
        
        print(f"📁 找到 {len(image_files)} 张图片")
        
        # 初始化GPU检测器
        detector = SimilarImageDetectorGPU(
            psnr_threshold=psnr_threshold,
            max_gpu_memory_mb=max_gpu_memory_mb,
            max_cpu_memory_mb=max_cpu_memory_mb,
            use_gpu=True,
            min_frame_distance=9,  # 设置严格的帧间距要求
            adjacent_frame_threshold=6  # 6帧内认定为相似
        )
        
        # 执行处理
        start_time = time.time()
        unique_images, similar_pairs = detector.auto_deduplicate(
            image_files, output_dir, similar_dir
        )
        end_time = time.time()
        
        # 生成报告
        report_file = r"F:\test\gpu_similarity_report.txt"
        detector.generate_report(report_file, image_files)
        
        print(f"\n🎉 GPU加速处理完成!")
        print(f"   - 处理时间: {end_time - start_time:.2f} 秒")
        print(f"   - 唯一图片目录: {output_dir}")
        print(f"   - 相似图片目录: {similar_dir}")
        print(f"   - 检测报告: {report_file}")
        print(f"   - GPU操作: {detector.stats['gpu_operations']}")
        print(f"   - CPU操作: {detector.stats['cpu_operations']}")
        print(f"   - 相邻帧过滤: {detector.stats['adjacent_frames_filtered']} 对")
        print(f"   - 8帧内相似: {detector.stats['adjacent_frames_similar']} 对")
    else:
        main()
