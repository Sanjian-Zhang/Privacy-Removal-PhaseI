#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理qualified目录下的图片 - 超严格相似图片检测
专门针对已经筛选过的高质量图片进行进一步去重
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Tuple, Dict, List
import shutil
import json
from datetime import datetime
import argparse

class QualifiedImageProcessor:
    """专门处理qualified图片的超严格检测器"""
    
    def __init__(self, 
                 psnr_threshold=65.0,        # 超高PSNR要求
                 ssim_threshold=0.92,        # 超高SSIM要求
                 composite_threshold=82.0,   # 超高复合评分
                 hist_threshold=0.95,        # 超严格直方图阈值
                 edge_threshold=0.88,        # 边缘相似度阈值
                 conditions_required=4):      # 要求满足4/5个条件
        
        self.psnr_threshold = psnr_threshold
        self.ssim_threshold = ssim_threshold  
        self.composite_threshold = composite_threshold
        self.hist_threshold = hist_threshold
        self.edge_threshold = edge_threshold
        self.conditions_required = conditions_required
        
        # GPU检测
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_comparisons': 0,
            'similar_found': 0,
            'final_kept': 0,
            'processing_time': 0
        }
    
    def calculate_psnr_gpu(self, img1, img2):
        """GPU加速PSNR计算"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转换为torch tensor
        img1_tensor = torch.from_numpy(img1.astype(np.float32)).to(self.device)
        img2_tensor = torch.from_numpy(img2.astype(np.float32)).to(self.device)
        
        # 计算MSE
        mse = torch.mean((img1_tensor - img2_tensor) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.cpu().item()
    
    def calculate_advanced_ssim(self, img1, img2):
        """高级SSIM计算（多尺度）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转换为灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)
        
        # 多尺度SSIM
        ssim_values = []
        scales = [1.0, 0.5, 0.25]  # 不同尺度
        
        for scale in scales:
            if scale < 1.0:
                h, w = int(gray1.shape[0] * scale), int(gray1.shape[1] * scale)
                g1 = cv2.resize(gray1, (w, h))
                g2 = cv2.resize(gray2, (w, h))
            else:
                g1, g2 = gray1, gray2
            
            # 计算单尺度SSIM
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_values.append(np.mean(ssim_map))
        
        # 加权平均（原尺度权重更大）
        weights = [0.6, 0.3, 0.1]
        return np.average(ssim_values, weights=weights)
    
    def calculate_color_histogram_similarity(self, img1, img2):
        """高精度彩色直方图相似度"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        similarities = []
        
        # RGB直方图（精细化）
        for i in range(3):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            # 标准化
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # 多种比较方法
            correl = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
            bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            # 综合评分
            chi_square_norm = 1.0 / (1.0 + chi_square)  # 转换为相似度
            bhattacharyya_norm = 1.0 - bhattacharyya
            
            channel_similarity = (correl * 0.4 + intersection * 0.3 + 
                                 chi_square_norm * 0.2 + bhattacharyya_norm * 0.1)
            similarities.append(max(0, channel_similarity))
        
        # HSV空间
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # 2D直方图（色调-饱和度）
        hist_hsv1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_hsv2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        hsv_similarity = cv2.compareHist(hist_hsv1, hist_hsv2, cv2.HISTCMP_CORREL)
        
        # Lab色彩空间
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        hist_lab1 = cv2.calcHist([lab1], [1, 2], None, [256, 256], [0, 256, 0, 256])
        hist_lab2 = cv2.calcHist([lab2], [1, 2], None, [256, 256], [0, 256, 0, 256])
        
        lab_similarity = cv2.compareHist(hist_lab1, hist_lab2, cv2.HISTCMP_CORREL)
        
        # 综合RGB、HSV、Lab
        rgb_avg = np.mean(similarities)
        final_similarity = (rgb_avg * 0.5 + hsv_similarity * 0.3 + lab_similarity * 0.2)
        
        return max(0, final_similarity)
    
    def calculate_edge_similarity(self, img1, img2):
        """高精度边缘相似度计算"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        edge_similarities = []
        
        # Canny边缘检测（多参数）
        canny_params = [(50, 150), (30, 100), (70, 200)]
        for low, high in canny_params:
            edges1 = cv2.Canny(gray1, low, high)
            edges2 = cv2.Canny(gray2, low, high)
            
            # 边缘相似度
            edge_diff = np.abs(edges1.astype(np.float32) - edges2.astype(np.float32))
            similarity = 1.0 - (np.mean(edge_diff) / 255.0)
            edge_similarities.append(similarity)
        
        # Sobel边缘检测
        sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        sobel1 = np.sqrt(sobelx1**2 + sobely1**2)
        
        sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        sobel2 = np.sqrt(sobelx2**2 + sobely2**2)
        
        # Sobel相似度
        sobel_diff = np.abs(sobel1 - sobel2)
        sobel_similarity = 1.0 - (np.mean(sobel_diff) / np.max([np.max(sobel1), np.max(sobel2)]))
        
        # 综合边缘相似度
        canny_avg = np.mean(edge_similarities)
        final_edge_similarity = (canny_avg * 0.7 + sobel_similarity * 0.3)
        
        return max(0, final_edge_similarity)
    
    def calculate_texture_similarity(self, img1, img2):
        """纹理相似度计算（LBP）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 简化的LBP计算
        def lbp_histogram(image):
            # 简单的局部二值模式
            h, w = image.shape
            lbp_image = np.zeros_like(image)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp_image[i, j] = code
            
            # 计算直方图
            hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
            return hist
        
        hist1 = lbp_histogram(gray1)
        hist2 = lbp_histogram(gray2)
        
        # 标准化
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        hist1 = hist1 / (np.sum(hist1) + 1e-10)
        hist2 = hist2 / (np.sum(hist2) + 1e-10)
        
        # 计算相似度
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, correlation)
    
    def are_images_ultra_similar(self, img_path1, img_path2):
        """超严格相似度判定（针对qualified图片）"""
        try:
            # 读取图像
            img1 = cv2.imread(str(img_path1))
            img2 = cv2.imread(str(img_path2))
            
            if img1 is None or img2 is None:
                return False, 0, {}
            
            # 1. 超高PSNR检测
            psnr = self.calculate_psnr_gpu(img1, img2)
            is_similar_psnr = psnr > self.psnr_threshold
            
            # 2. 多尺度SSIM检测
            ssim = self.calculate_advanced_ssim(img1, img2)
            is_similar_ssim = ssim > self.ssim_threshold
            
            # 3. 高精度直方图相似度
            hist_sim = self.calculate_color_histogram_similarity(img1, img2)
            is_similar_hist = hist_sim > self.hist_threshold
            
            # 4. 边缘相似度
            edge_sim = self.calculate_edge_similarity(img1, img2)
            is_similar_edge = edge_sim > self.edge_threshold
            
            # 5. 纹理相似度
            texture_sim = self.calculate_texture_similarity(img1, img2)
            is_similar_texture = texture_sim > 0.85
            
            # 6. 超严格复合评分
            psnr_normalized = min(psnr / 75.0, 1.0)  # 基准提升到75
            composite_score = (psnr_normalized * 0.25 + ssim * 0.25 + 
                              hist_sim * 0.2 + edge_sim * 0.2 + texture_sim * 0.1) * 100
            is_similar_composite = composite_score > self.composite_threshold
            
            # 超严格条件检查
            conditions = [is_similar_psnr, is_similar_ssim, is_similar_hist, 
                         is_similar_edge, is_similar_composite]
            conditions_met = sum(conditions)
            
            # 必须满足核心条件且达到要求数量
            is_similar = (conditions_met >= self.conditions_required and 
                         is_similar_psnr and is_similar_ssim and is_similar_hist)
            
            details = {
                'psnr': psnr,
                'ssim': ssim,
                'hist_similarity': hist_sim,
                'edge_similarity': edge_sim,
                'texture_similarity': texture_sim,
                'composite_score': composite_score,
                'conditions_met': f"{conditions_met}/5",
                'thresholds': {
                    'psnr_threshold': self.psnr_threshold,
                    'ssim_threshold': self.ssim_threshold,
                    'hist_threshold': self.hist_threshold,
                    'edge_threshold': self.edge_threshold,
                    'composite_threshold': self.composite_threshold
                },
                'is_similar_breakdown': {
                    'psnr': is_similar_psnr,
                    'ssim': is_similar_ssim,
                    'histogram': is_similar_hist,
                    'edge': is_similar_edge,
                    'texture': is_similar_texture,
                    'composite': is_similar_composite
                }
            }
            
            self.stats['total_comparisons'] += 1
            if is_similar:
                self.stats['similar_found'] += 1
                
            return is_similar, composite_score, details
            
        except Exception as e:
            print(f"❌ 比较失败 {img_path1} vs {img_path2}: {e}")
            return False, 0, {}
    
    def select_best_qualified_image(self, image_group):
        """为qualified图片选择最佳代表"""
        best_img = None
        best_score = 0
        
        print(f"  🔍 评估 {len(image_group)} 张相似图片...")
        
        for img_path in image_group:
            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # 文件大小权重
                file_size = img_path.stat().st_size
                size_score = min(file_size / 500000, 1.0)  # 500KB基准
                
                # 图像清晰度（拉普拉斯方差）
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                clarity_score = min(clarity / 500, 1.0)  # 500基准
                
                # 分辨率权重
                height, width = img.shape[:2]
                resolution_score = min((width * height) / 8000000, 1.0)  # 8MP基准
                
                # 图像对比度
                contrast = np.std(gray)
                contrast_score = min(contrast / 50, 1.0)  # 50基准
                
                # 边缘密度
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (width * height)
                edge_score = min(edge_density * 10, 1.0)
                
                # 综合评分（优化权重）
                score = (clarity_score * 0.35 +      # 清晰度最重要
                        contrast_score * 0.25 +      # 对比度
                        edge_score * 0.2 +           # 边缘信息
                        resolution_score * 0.15 +    # 分辨率
                        size_score * 0.05)           # 文件大小
                
                print(f"    📊 {img_path.name}: 清晰度={clarity:.0f}, 对比度={contrast:.1f}, "
                      f"边缘密度={edge_density:.3f}, 总分={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_img = img_path
                    
            except Exception as e:
                print(f"⚠️ 评估图片失败 {img_path}: {e}")
                continue
        
        if best_img:
            print(f"  ✅ 最佳图片: {best_img.name} (分数: {best_score:.3f})")
        
        return best_img or image_group[0]
    
    def process_qualified_directory(self, qualified_dir, output_dir=None):
        """处理qualified目录"""
        qualified_path = Path(qualified_dir)
        if not qualified_path.exists():
            print(f"❌ 目录不存在: {qualified_path}")
            return
        
        # 获取所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in qualified_path.glob("*") if f.suffix.lower() in image_extensions]
        
        if len(image_files) < 2:
            print("📁 图片数量不足，无需检测")
            return
        
        self.stats['total_images'] = len(image_files)
        
        print(f"🎯 开始超严格处理 qualified 目录: {qualified_path}")
        print(f"📊 输入图片数量: {len(image_files)}")
        print(f"⚙️ 超严格阈值设置:")
        print(f"   - PSNR: {self.psnr_threshold}")
        print(f"   - SSIM: {self.ssim_threshold}")
        print(f"   - 直方图: {self.hist_threshold}")
        print(f"   - 边缘: {self.edge_threshold}")
        print(f"   - 复合评分: {self.composite_threshold}")
        print(f"   - 条件要求: {self.conditions_required}/5")
        print()
        
        start_time = datetime.now()
        
        duplicate_groups = []
        processed = set()
        
        for i, img1 in enumerate(image_files):
            if str(img1) in processed:
                continue
                
            current_group = [img1]
            processed.add(str(img1))
            
            print(f"🔍 检测图片 {i+1}/{len(image_files)}: {img1.name}")
            
            for j, img2 in enumerate(image_files[i+1:], i+1):
                if str(img2) in processed:
                    continue
                    
                is_similar, score, details = self.are_images_ultra_similar(img1, img2)
                
                if is_similar:
                    print(f"  🎯 发现超相似: {img2.name}")
                    print(f"     📊 PSNR: {details['psnr']:.1f}, SSIM: {details['ssim']:.3f}, "
                          f"直方图: {details['hist_similarity']:.3f}")
                    print(f"     📊 边缘: {details['edge_similarity']:.3f}, "
                          f"复合分: {details['composite_score']:.1f}")
                    current_group.append(img2)
                    processed.add(str(img2))
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                print(f"  📁 相似组大小: {len(current_group)}")
            
            print()
        
        # 处理结果
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        if not duplicate_groups:
            print("🎉 未发现超相似图片！所有图片都足够独特。")
            self.stats['final_kept'] = len(image_files)
            
            # 如果指定输出目录，复制所有文件
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                for img in image_files:
                    shutil.copy2(img, output_path / img.name)
                print(f"📁 所有图片已复制到: {output_path}")
            return
        
        print(f"🎯 发现 {len(duplicate_groups)} 个超相似组:")
        
        # 创建输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = qualified_path.parent / "qualified_final"
            output_path.mkdir(parents=True, exist_ok=True)
        
        kept_files = []
        removed_files = []
        
        # 处理相似组
        for i, group in enumerate(duplicate_groups):
            print(f"\n📁 处理相似组 {i+1}/{len(duplicate_groups)}: {len(group)} 张图片")
            
            # 选择最佳图片
            best_img = self.select_best_qualified_image(group)
            
            for img in group:
                if img == best_img:
                    kept_files.append(img)
                    shutil.copy2(img, output_path / img.name)
                    print(f"  ✅ 保留: {img.name}")
                else:
                    removed_files.append(img)
                    print(f"  🗑️ 移除: {img.name}")
        
        # 复制非重复文件
        for img in image_files:
            if not any(img in group for group in duplicate_groups):
                shutil.copy2(img, output_path / img.name)
                kept_files.append(img)
        
        self.stats['final_kept'] = len(kept_files)
        
        # 生成详细报告
        report = {
            'timestamp': end_time.isoformat(),
            'input_directory': str(qualified_path),
            'output_directory': str(output_path),
            'ultra_strict_settings': {
                'psnr_threshold': self.psnr_threshold,
                'ssim_threshold': self.ssim_threshold,
                'composite_threshold': self.composite_threshold,
                'hist_threshold': self.hist_threshold,
                'edge_threshold': self.edge_threshold,
                'conditions_required': self.conditions_required
            },
            'statistics': self.stats,
            'results': {
                'total_input_images': len(image_files),
                'duplicate_groups_found': len(duplicate_groups),
                'images_kept': len(kept_files),
                'images_removed': len(removed_files),
                'reduction_rate': f"{(len(removed_files)/len(image_files)*100):.1f}%"
            },
            'duplicate_groups': [
                {
                    'group_id': i+1,
                    'images': [str(img) for img in group],
                    'kept_image': str(self.select_best_qualified_image(group))
                }
                for i, group in enumerate(duplicate_groups)
            ]
        }
        
        report_path = output_path / "ultra_strict_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 超严格处理完成!")
        print(f"📊 处理统计:")
        print(f"   ⏱️  处理时间: {self.stats['processing_time']:.1f}秒")
        print(f"   📁 输入图片: {len(image_files)}张")
        print(f"   🔍 比较次数: {self.stats['total_comparisons']}")
        print(f"   🎯 相似组数: {len(duplicate_groups)}")
        print(f"   ✅ 保留图片: {len(kept_files)}张")
        print(f"   🗑️ 移除图片: {len(removed_files)}张")
        print(f"   📉 压缩率: {(len(removed_files)/len(image_files)*100):.1f}%")
        print(f"   📁 输出目录: {output_path}")
        print(f"   📋 详细报告: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qualified图片超严格处理器")
    parser.add_argument('qualified_dir', help='qualified图片目录路径')
    parser.add_argument('-o', '--output', help='输出目录（筛选后的最终图片）')
    parser.add_argument('--psnr', type=float, default=65.0, help='PSNR阈值 (默认: 65.0)')
    parser.add_argument('--ssim', type=float, default=0.92, help='SSIM阈值 (默认: 0.92)')
    parser.add_argument('--composite', type=float, default=82.0, help='复合评分阈值 (默认: 82.0)')
    parser.add_argument('--hist', type=float, default=0.95, help='直方图阈值 (默认: 0.95)')
    parser.add_argument('--edge', type=float, default=0.88, help='边缘阈值 (默认: 0.88)')
    parser.add_argument('--conditions', type=int, default=4, help='必须满足的条件数 (默认: 4/5)')
    
    args = parser.parse_args()
    
    processor = QualifiedImageProcessor(
        psnr_threshold=args.psnr,
        ssim_threshold=args.ssim,
        composite_threshold=args.composite,
        hist_threshold=args.hist,
        edge_threshold=args.edge,
        conditions_required=args.conditions
    )
    
    processor.process_qualified_directory(args.qualified_dir, args.output)

if __name__ == "__main__":
    main()
