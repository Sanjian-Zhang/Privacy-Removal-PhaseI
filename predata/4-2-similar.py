#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Tuple, Dict, List
import shutil
import json
from datetime import datetime

class StrictSimilarityDetector:
    """超严格相似图片检测器 - 大幅减少误判"""
    
    def __init__(self, 
                 psnr_threshold=60.0,        # 提高到60（原50）
                 ssim_threshold=0.85,        # 提高到0.85（原0.7）
                 composite_threshold=75.0,   # 提高到75（原65）
                 hist_threshold=0.9,         # 新增直方图严格阈值
                 conditions_required=3):      # 要求满足3个条件（原2个）
        
        self.psnr_threshold = psnr_threshold
        self.ssim_threshold = ssim_threshold  
        self.composite_threshold = composite_threshold
        self.hist_threshold = hist_threshold
        self.conditions_required = conditions_required
        
        # GPU检测
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 统计信息
        self.stats = {
            'total_comparisons': 0,
            'similar_found': 0,
            'strict_rejections': 0,
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
    
    def calculate_ssim_opencv(self, img1, img2):
        """使用OpenCV计算SSIM（更准确）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转换为灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 使用高斯窗口计算SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def calculate_histogram_similarity_strict(self, img1, img2):
        """更严格的直方图相似度计算"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 多通道直方图比较
        similarities = []
        
        # RGB各通道直方图
        for i in range(3):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            # 标准化
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # 使用巴氏距离（更严格）
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            similarities.append(1.0 - similarity)  # 转换为相似度
        
        # HSV空间直方图
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        hist_hsv1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_hsv2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        hsv_similarity = cv2.compareHist(hist_hsv1, hist_hsv2, cv2.HISTCMP_CORREL)
        
        # 综合评分
        rgb_avg = np.mean(similarities)
        final_similarity = (rgb_avg * 0.6 + hsv_similarity * 0.4)
        
        return max(0, final_similarity)
    
    def calculate_structural_difference(self, img1, img2):
        """计算结构差异度（边缘检测对比）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 转灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Canny边缘检测
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # 计算边缘相似度
        edge_diff = np.abs(edges1.astype(np.float32) - edges2.astype(np.float32))
        edge_similarity = 1.0 - (np.mean(edge_diff) / 255.0)
        
        return edge_similarity
    
    def quick_similarity_check(self, img_path1, img_path2):
        """快速预检查 - 避免不必要的详细计算"""
        try:
            # 文件大小快速检查
            size1 = img_path1.stat().st_size
            size2 = img_path2.stat().st_size
            size_ratio = min(size1, size2) / max(size1, size2)
            
            # 如果文件大小差异超过50%，不太可能是相似图片
            if size_ratio < 0.5:
                return False
            
            # 读取图像并快速检查分辨率
            img1 = cv2.imread(str(img_path1))
            img2 = cv2.imread(str(img_path2))
            
            if img1 is None or img2 is None:
                return False
            
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # 分辨率差异检查
            res_ratio = min(w1*h1, w2*h2) / max(w1*h1, w2*h2)
            if res_ratio < 0.7:  # 分辨率差异超过30%
                return False
            
            # 快速直方图检查（只检查一个通道）
            hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])  # 降低精度
            hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # 快速相关性检查
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 如果连基本的直方图相关性都很低，跳过详细检查
            return correlation > 0.6
            
        except Exception:
            return True  # 出错时继续详细检查
    
    def are_images_strictly_similar(self, img_path1, img_path2):
        """超严格相似度判定 - 优化版本"""
        try:
            # 🚀 快速预检查
            if not self.quick_similarity_check(img_path1, img_path2):
                return False, 0, {}
            
            # 读取图像
            img1 = cv2.imread(str(img_path1))
            img2 = cv2.imread(str(img_path2))
            
            if img1 is None or img2 is None:
                return False, 0, {}
            
            # 1. PSNR检测
            psnr = self.calculate_psnr_gpu(img1, img2)
            is_similar_psnr = psnr > self.psnr_threshold
            
            # 2. SSIM检测（更严格）
            ssim = self.calculate_ssim_opencv(img1, img2)
            is_similar_ssim = ssim > self.ssim_threshold
            
            # 3. 直方图相似度（更严格）
            hist_sim = self.calculate_histogram_similarity_strict(img1, img2)
            is_similar_hist = hist_sim > self.hist_threshold
            
            # 4. 结构相似度（新增）
            struct_sim = self.calculate_structural_difference(img1, img2)
            is_similar_struct = struct_sim > 0.8
            
            # 5. 复合评分（更严格权重）
            psnr_normalized = min(psnr / 70.0, 1.0)  # 提高基准到70
            composite_score = (psnr_normalized * 0.3 + ssim * 0.3 + hist_sim * 0.25 + struct_sim * 0.15) * 100
            is_similar_composite = composite_score > self.composite_threshold
            
            # 严格条件检查
            conditions = [is_similar_psnr, is_similar_ssim, is_similar_hist, is_similar_struct, is_similar_composite]
            conditions_met = sum(conditions)
            
            # 必须满足所有主要条件
            is_similar = conditions_met >= self.conditions_required and is_similar_psnr and is_similar_ssim
            
            details = {
                'psnr': psnr,
                'ssim': ssim,
                'hist_similarity': hist_sim,
                'structural_similarity': struct_sim,
                'composite_score': composite_score,
                'conditions_met': f"{conditions_met}/5",
                'is_similar_breakdown': {
                    'psnr': is_similar_psnr,
                    'ssim': is_similar_ssim,
                    'histogram': is_similar_hist,
                    'structural': is_similar_struct,
                    'composite': is_similar_composite
                }
            }
            
            self.stats['total_comparisons'] += 1
            if is_similar:
                self.stats['similar_found'] += 1
            else:
                self.stats['strict_rejections'] += 1
                
            return is_similar, composite_score, details
            
        except Exception as e:
            print(f"❌ 比较失败 {img_path1} vs {img_path2}: {e}")
            return False, 0, {}
    
    def extract_frame_number(self, filename):
        """从文件名中提取帧编号"""
        import re
        # 匹配 video40_frame_000xxx.jpg 格式
        match = re.search(r'video\d+_frame_(\d+)\.jpg', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def detect_and_remove_strict_duplicates(self, directory_path, output_dir=None, nearby_range=5):
        """严格检测并移除重复图片 - 只比较相近编号的图片"""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ 目录不存在: {directory}")
            return
        
        # 获取所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in directory.glob("*") if f.suffix.lower() in image_extensions]
        
        if len(image_files) < 2:
            print("📁 图片数量不足，无需检测")
            return
        
        # 按帧编号排序
        image_files.sort(key=lambda x: self.extract_frame_number(x.name))
        
        # 显示帧编号范围
        frame_numbers = [self.extract_frame_number(f.name) for f in image_files]
        print(f"🔍 智能检测 {len(image_files)} 张图片 (帧编号: {min(frame_numbers)}-{max(frame_numbers)})")
        print(f"🎯 只比较相近 ±{nearby_range} 帧内的图片")
        
        # 计算实际比较次数
        total_comparisons = 0
        for i in range(len(image_files)):
            for j in range(i+1, min(i+1+nearby_range, len(image_files))):
                total_comparisons += 1
        
        print(f"📊 智能比较次数: {total_comparisons} (相比全比较节省: {((len(image_files)*(len(image_files)-1)//2 - total_comparisons)/(len(image_files)*(len(image_files)-1)//2)*100):.1f}%)")
        print(f"⚙️ 当前阈值 - PSNR: {self.psnr_threshold}, SSIM: {self.ssim_threshold}, 复合: {self.composite_threshold}")
        print(f"🎯 要求满足 {self.conditions_required}/5 个条件")
        print()
        
        duplicate_groups = []
        processed = set()
        comparison_count = 0
        
        for i, img1 in enumerate(image_files):
            if str(img1) in processed:
                continue
                
            current_group = [img1]
            processed.add(str(img1))
            
            frame_num1 = self.extract_frame_number(img1.name)
            print(f"🔍 检测图片 {i+1}/{len(image_files)}: {img1.name} (帧号: {frame_num1})")
            
            # 只比较相近的图片
            for j in range(i+1, min(i+1+nearby_range, len(image_files))):
                img2 = image_files[j]
                if str(img2) in processed:
                    continue
                
                frame_num2 = self.extract_frame_number(img2.name)
                frame_diff = abs(frame_num2 - frame_num1)
                
                comparison_count += 1
                
                # 显示比较信息
                print(f"  📊 比较 {img2.name} (帧差: {frame_diff}) [{comparison_count}/{total_comparisons}]", end="")
                
                is_similar, score, details = self.are_images_strictly_similar(img1, img2)
                
                if is_similar:
                    print(f" ✅ 相似!")
                    print(f"     📊 PSNR: {details['psnr']:.1f}, SSIM: {details['ssim']:.3f}")
                    print(f"     📊 复合分: {details['composite_score']:.1f}, 条件: {details['conditions_met']}")
                    current_group.append(img2)
                    processed.add(str(img2))
                else:
                    print(f" ❌ 不同")
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                print(f"  📁 相似组大小: {len(current_group)}")
            
            # 显示进度
            progress = ((i + 1) / len(image_files)) * 100
            print(f"  📊 总进度: {i+1}/{len(image_files)} ({progress:.1f}%)")
            print()
        
        print(f"📊 智能比较完成! 实际比较: {comparison_count}/{total_comparisons}")
        
        # 处理重复组
        if not duplicate_groups:
            print("✅ 未发现相似的相邻帧图片！所有图片都足够独特。")
            
            # 如果指定输出目录，复制所有文件
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                for img in image_files:
                    shutil.copy2(img, output_path / img.name)
                print(f"📁 所有 {len(image_files)} 张图片已复制到: {output_path}")
            return
        
        print(f"\n🎯 发现 {len(duplicate_groups)} 个相似的相邻帧组:")
        
        # 创建输出目录
        output_path = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        kept_files = []
        removed_files = []
        
        for i, group in enumerate(duplicate_groups):
            frame_numbers_in_group = [self.extract_frame_number(img.name) for img in group]
            print(f"\n📁 处理相似组 {i+1}/{len(duplicate_groups)}: {len(group)} 张相邻帧")
            print(f"   📊 帧编号范围: {min(frame_numbers_in_group)}-{max(frame_numbers_in_group)}")
            
            # 选择最佳图片
            best_img = self.select_best_image_strict(group)
            
            for img in group:
                frame_num = self.extract_frame_number(img.name)
                if img == best_img:
                    kept_files.append(img)
                    if output_path:
                        shutil.copy2(img, output_path / img.name)
                    print(f"  ✅ 保留: {img.name} (帧号: {frame_num})")
                else:
                    removed_files.append(img)
                    print(f"  🗑️ 移除: {img.name} (帧号: {frame_num})")
        
        # 复制非重复文件
        if output_path:
            for img in image_files:
                if not any(img in group for group in duplicate_groups):
                    shutil.copy2(img, output_path / img.name)
                    kept_files.append(img)
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(directory),
            'output_directory': str(output_path) if output_path else None,
            'detection_method': 'nearby_frames_only',
            'nearby_range': nearby_range,
            'detection_settings': {
                'psnr_threshold': self.psnr_threshold,
                'ssim_threshold': self.ssim_threshold,
                'composite_threshold': self.composite_threshold,
                'hist_threshold': self.hist_threshold,
                'conditions_required': self.conditions_required
            },
            'statistics': self.stats,
            'frame_analysis': {
                'frame_range': f"{min(frame_numbers)}-{max(frame_numbers)}",
                'total_frames': len(image_files),
                'comparisons_made': comparison_count,
                'comparisons_saved': (len(image_files)*(len(image_files)-1)//2) - comparison_count,
                'efficiency_gain': f"{((len(image_files)*(len(image_files)-1)//2 - comparison_count)/(len(image_files)*(len(image_files)-1)//2)*100):.1f}%"
            },
            'results': {
                'total_images': len(image_files),
                'total_comparisons': comparison_count,
                'duplicate_groups': len(duplicate_groups),
                'images_kept': len(kept_files),
                'images_removed': len(removed_files),
                'reduction_rate': f"{(len(removed_files)/len(image_files)*100):.1f}%"
            },
            'duplicate_groups': [
                {
                    'group_id': i+1,
                    'images': [str(img) for img in group],
                    'frame_numbers': [self.extract_frame_number(img.name) for img in group],
                    'kept_image': str(self.select_best_image_strict(group))
                }
                for i, group in enumerate(duplicate_groups)
            ]
        }
        
        report_path = output_path / "nearby_frames_similarity_report.json" if output_path else Path(directory) / "nearby_frames_similarity_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 智能相邻帧筛选完成!")
        print(f"   📊 总图片数: {len(image_files)}")
        print(f"   🔍 实际比较: {comparison_count}")
        print(f"   ⚡ 效率提升: {((len(image_files)*(len(image_files)-1)//2 - comparison_count)/(len(image_files)*(len(image_files)-1)//2)*100):.1f}%")
        print(f"   📁 相似组数: {len(duplicate_groups)}")  
        print(f"   ✅ 保留图片: {len(kept_files)}")
        print(f"   🗑️ 移除图片: {len(removed_files)}")
        print(f"   📉 压缩率: {(len(removed_files)/len(image_files)*100):.1f}%")
        print(f"   📋 报告保存: {report_path}")
        
        if output_path:
            print(f"   📁 筛选结果: {output_path}")
            
        # 显示帧分布统计
        if duplicate_groups:
            print(f"\n📊 相似帧分布:")
            for i, group in enumerate(duplicate_groups):
                frame_nums = [self.extract_frame_number(img.name) for img in group]
                print(f"   组{i+1}: 帧{min(frame_nums)}-{max(frame_nums)} ({len(group)}张)")
        
        print(f"\n💡 建议: 如需更严格筛选，可调整相邻范围参数 (当前: ±{nearby_range})")
    
    def select_best_image_strict(self, image_group):
        """严格选择最佳图片 - 针对qualified图片优化"""
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
                
                # 颜色丰富度（饱和度）
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                saturation_score = min(saturation / 100, 1.0)
                
                # 综合评分（针对qualified图片优化权重）
                score = (clarity_score * 0.4 +         # 清晰度最重要
                        contrast_score * 0.2 +         # 对比度
                        edge_score * 0.15 +            # 边缘信息
                        resolution_score * 0.15 +      # 分辨率
                        saturation_score * 0.05 +      # 颜色丰富度
                        size_score * 0.05)             # 文件大小
                
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

def process_qualified_images():
    """专门处理qualified目录的图片 - 智能相邻帧比较"""
    # 固定的qualified目录路径
    qualified_dir = "/home/zhiqics/sanjian/predata/output_frames40/qualified"
    output_dir = "/home/zhiqics/sanjian/predata/output_frames40/final_unique"
    
    print("🎯 专门处理qualified图片 - 智能相邻帧模式")
    print(f"📁 输入目录: {qualified_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("💡 只比较帧编号相近的图片，大幅提升处理速度")
    print()
    
    # 检查目录是否存在
    if not Path(qualified_dir).exists():
        print(f"❌ 目录不存在: {qualified_dir}")
        return
    
    # 使用超严格参数（针对qualified图片）
    detector = StrictSimilarityDetector(
        psnr_threshold=65.0,        # 超高PSNR要求
        ssim_threshold=0.9,         # 超高SSIM要求
        composite_threshold=80.0,   # 超高复合评分
        hist_threshold=0.95,        # 超严格直方图阈值
        conditions_required=4       # 要求满足4/5个条件
    )
    
    print("⚙️ 超严格参数设置:")
    print(f"   - PSNR阈值: {detector.psnr_threshold}")
    print(f"   - SSIM阈值: {detector.ssim_threshold}")
    print(f"   - 复合评分: {detector.composite_threshold}")
    print(f"   - 直方图阈值: {detector.hist_threshold}")
    print(f"   - 条件要求: {detector.conditions_required}/5")
    print(f"   - 相邻范围: ±5 帧")
    print()
    
    # 开始智能处理
    detector.detect_and_remove_strict_duplicates(qualified_dir, output_dir, nearby_range=5)

def main():
    """主函数 - 支持交互模式和命令行模式"""
    import argparse
    import sys
    
    # 如果没有命令行参数，直接处理qualified目录
    if len(sys.argv) == 1:
        process_qualified_images()
        return
    
    # 命令行模式
    parser = argparse.ArgumentParser(description="超严格相似图片检测器 - 智能相邻帧模式")
    parser.add_argument('input_dir', nargs='?', help='输入图片目录（可选，默认处理qualified目录）')
    parser.add_argument('-o', '--output', help='输出目录（筛选后的图片）')
    parser.add_argument('--psnr', type=float, default=65.0, help='PSNR阈值 (默认: 65.0)')
    parser.add_argument('--ssim', type=float, default=0.9, help='SSIM阈值 (默认: 0.9)')
    parser.add_argument('--composite', type=float, default=80.0, help='复合评分阈值 (默认: 80.0)')
    parser.add_argument('--hist', type=float, default=0.95, help='直方图阈值 (默认: 0.95)')
    parser.add_argument('--conditions', type=int, default=4, help='必须满足的条件数 (默认: 4/5)')
    parser.add_argument('--range', type=int, default=5, help='相邻帧比较范围 (默认: 5)')
    parser.add_argument('--qualified', action='store_true', help='直接处理qualified目录')
    
    args = parser.parse_args()
    
    # 如果指定处理qualified或没有输入目录，处理qualified
    if args.qualified or not args.input_dir:
        process_qualified_images()
        return
    
    # 处理指定目录
    detector = StrictSimilarityDetector(
        psnr_threshold=args.psnr,
        ssim_threshold=args.ssim,
        composite_threshold=args.composite,
        hist_threshold=args.hist,
        conditions_required=args.conditions
    )
    
    detector.detect_and_remove_strict_duplicates(args.input_dir, args.output, nearby_range=args.range)

if __name__ == "__main__":
    main()