#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速度优化测试脚本
测试不影响图片质量的前提下的速度优化效果
"""

import os
import time
import shutil
from face_plate_classifier_improved import Config, FacePlateClassifier

def create_test_data():
    """创建测试数据：复制一些图片进行测试"""
    source_dir = "/home/zhiqics/sanjian/predata/frames38"
    test_dir = "/tmp/speed_test_input"
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # 复制前10张图片，并创建一些重复文件来测试去重功能
    source_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
    
    copied_count = 0
    for i, filename in enumerate(source_files):
        source_path = os.path.join(source_dir, filename)
        
        # 复制原文件
        target_path = os.path.join(test_dir, filename)
        shutil.copy2(source_path, target_path)
        copied_count += 1
        
        # 创建重复文件（不同名称但内容相同）
        if i < 3:
            duplicate_name = f"duplicate_{i}_{filename}"
            duplicate_path = os.path.join(test_dir, duplicate_name)
            shutil.copy2(source_path, duplicate_path)
            copied_count += 1
        
        # 创建一些小文件
        if i < 2:
            small_file_path = os.path.join(test_dir, f"small_{i}.jpg")
            with open(small_file_path, 'wb') as f:
                f.write(b"small file content")
            copied_count += 1
    
    print(f"✅ 创建了 {copied_count} 个测试文件")
    return test_dir

def speed_test_with_optimizations():
    """测试启用优化的速度"""
    print("🚀 测试启用速度优化的性能...")
    
    test_input_dir = create_test_data()
    test_output_dir = "/tmp/speed_test_output_optimized"
    
    class OptimizedTestConfig(Config):
        INPUT_DIR = test_input_dir
        OUTPUT_BASE_DIR = test_output_dir
        # 启用所有优化
        ENABLE_SMART_SKIP = True
        SKIP_PROCESSED_FILES = True
        ENABLE_FAST_PREPROCESSING = True
        ENABLE_RESULT_CACHE = True
        EARLY_STOP_ON_SCORE = True
        MIN_FILE_SIZE = 1024  # 跳过小于1KB的文件
    
    config = OptimizedTestConfig()
    
    start_time = time.time()
    
    try:
        classifier = FacePlateClassifier(config)
        classifier.run()
        
        processing_time = time.time() - start_time
        
        print(f"\n📊 优化版本结果:")
        print(f"  ⏰ 总处理时间: {processing_time:.2f}秒")
        print(f"  🚀 平均速度: {len(os.listdir(test_input_dir))/processing_time:.2f} 张/秒")
        print(f"  ⚡ 跳过重复: {classifier.stats['skipped_duplicate']}")
        print(f"  📏 跳过小文件: {classifier.stats['skipped_small']}")
        print(f"  💾 缓存命中: {classifier.stats['cache_hits']}")
        
        return processing_time, classifier.stats
        
    finally:
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)

def speed_test_without_optimizations():
    """测试不启用优化的速度（用于对比）"""
    print("\n🐌 测试不启用速度优化的性能...")
    
    test_input_dir = create_test_data()
    test_output_dir = "/tmp/speed_test_output_basic"
    
    class BasicTestConfig(Config):
        INPUT_DIR = test_input_dir
        OUTPUT_BASE_DIR = test_output_dir
        # 禁用优化
        ENABLE_SMART_SKIP = False
        SKIP_PROCESSED_FILES = False
        ENABLE_FAST_PREPROCESSING = False
        ENABLE_RESULT_CACHE = False
        EARLY_STOP_ON_SCORE = False
        MIN_FILE_SIZE = 0
    
    config = BasicTestConfig()
    
    start_time = time.time()
    
    try:
        classifier = FacePlateClassifier(config)
        classifier.run()
        
        processing_time = time.time() - start_time
        
        print(f"\n📊 基础版本结果:")
        print(f"  ⏰ 总处理时间: {processing_time:.2f}秒")
        print(f"  🚀 平均速度: {len(os.listdir(test_input_dir))/processing_time:.2f} 张/秒")
        
        return processing_time, classifier.stats
        
    finally:
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)

def compare_performance():
    """对比性能"""
    print("🧪 开始速度优化效果测试...\n")
    
    # 测试优化版本
    opt_time, opt_stats = speed_test_with_optimizations()
    
    # 测试基础版本  
    basic_time, basic_stats = speed_test_without_optimizations()
    
    # 性能对比
    print("\n" + "="*60)
    print("📈 性能对比结果:")
    print("="*60)
    
    speedup = basic_time / opt_time if opt_time > 0 else float('inf')
    time_saved = basic_time - opt_time
    percentage_saved = (time_saved / basic_time) * 100 if basic_time > 0 else 0
    
    print(f"⏰ 基础版本时间: {basic_time:.2f}秒")
    print(f"⚡ 优化版本时间: {opt_time:.2f}秒")
    print(f"🚀 速度提升: {speedup:.2f}x")
    print(f"💾 节省时间: {time_saved:.2f}秒 ({percentage_saved:.1f}%)")
    
    print(f"\n📊 优化效果统计:")
    print(f"  🔄 重复文件检测: {opt_stats['skipped_duplicate']} 个")
    print(f"  📏 小文件过滤: {opt_stats['skipped_small']} 个")
    print(f"  💾 缓存命中: {opt_stats['cache_hits']} 次")
    
    # 清理测试数据
    for test_dir in ["/tmp/speed_test_input"]:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    
    print(f"\n✅ 测试完成，临时文件已清理")
    print("="*60)

if __name__ == "__main__":
    compare_performance()
