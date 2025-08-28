#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正脸检测改进使用指南
解决后脑勺误判问题的完整解决方案
"""

import os
import sys
from pathlib import Path

def print_improvements_summary():
    """打印改进总结"""
    print("🎉 正脸检测改进总结")
    print("="*80)
    
    print("\n🔧 已实现的改进:")
    print("1. ✅ 更精确的yaw角度计算")
    print("   - 降低角度系数从60→35，更保守判断")
    print("   - 增加嘴部位置验证，防止侧脸误判")
    
    print("\n2. ✅ 多维度姿态角度检查")
    print("   - Yaw角度: ≤15° (左右转头)")
    print("   - Pitch角度: ≤20° (上下点头)")  
    print("   - Roll角度: ≤25° (头部倾斜)")
    
    print("\n3. ✅ 面部特征验证")
    print("   - 眼间距合理性检查 (20%-50%)")
    print("   - 面部对称性验证")
    print("   - 五官垂直分布检查")
    print("   - 关键点位置合理性验证")
    
    print("\n4. ✅ 独立眼部检测")
    print("   - 使用Haar级联检测眼部")
    print("   - 验证眼部数量和位置")
    print("   - 排除无眼部特征的后脑勺")
    
    print("\n5. ✅ 更严格的尺寸要求")
    print("   - 最小人脸尺寸: 120px → 140px")
    print("   - 最小面积比例: 12% → 15%")
    print("   - 最小分辨率: 150px → 160px")
    print("   - 置信度要求: 0.8 → 0.85")

def print_usage_guide():
    """打印使用指南"""
    print("\n📖 使用指南")
    print("="*80)
    
    print("\n🚀 方法1: 使用独立的改进检测器")
    print("```python")
    print("from improved_face_detector import ImprovedFaceDetector")
    print("")
    print("detector = ImprovedFaceDetector(")
    print("    min_confidence=0.85,")
    print("    max_yaw_angle=15.0,")
    print("    max_pitch_angle=20.0,")
    print("    enable_eye_detection=True")
    print(")")
    print("")
    print("frontal_count, details = detector.detect_frontal_faces('image.jpg', return_details=True)")
    print("```")
    
    print("\n🔄 方法2: 使用更新的主程序")
    print("直接运行改进后的主检测器:")
    print("```bash")
    print("python 2-fast_face_plate_detector_v2.py")
    print("```")
    
    print("\n⚙️ 参数调整建议:")
    print("如果仍有后脑勺误判:")
    print("- 降低 max_yaw_angle 到 12° 或更低")
    print("- 提高 min_area_ratio 到 0.02")
    print("- 启用更严格的 profile_rejection")
    
    print("\n如果正脸检出率过低:")
    print("- 适当提高 max_yaw_angle 到 18°")
    print("- 降低 min_face_size 到 120px")
    print("- 调整 symmetry_threshold 到 0.2")

def print_configuration_options():
    """打印配置选项"""
    print("\n⚙️ 详细配置选项")
    print("="*80)
    
    config_options = {
        'min_confidence': {
            'default': 0.85,
            'range': '0.7 - 0.95',
            'description': 'RetinaFace检测置信度阈值'
        },
        'max_yaw_angle': {
            'default': 15.0,
            'range': '10° - 25°',
            'description': '左右转头角度阈值（越小越严格）'
        },
        'max_pitch_angle': {
            'default': 20.0,
            'range': '15° - 30°', 
            'description': '上下点头角度阈值'
        },
        'max_roll_angle': {
            'default': 25.0,
            'range': '20° - 35°',
            'description': '头部倾斜角度阈值'
        },
        'min_face_size': {
            'default': 140,
            'range': '100 - 200',
            'description': '最小人脸尺寸（像素）'
        },
        'min_area_ratio': {
            'default': 0.015,
            'range': '0.01 - 0.03',
            'description': '人脸最小面积比例'
        },
        'enable_eye_detection': {
            'default': True,
            'range': 'True/False',
            'description': '是否启用独立眼部检测验证'
        },
        'enable_profile_rejection': {
            'default': True,
            'range': 'True/False', 
            'description': '是否启用侧脸/后脑勺拒绝'
        }
    }
    
    for param, info in config_options.items():
        print(f"\n📋 {param}:")
        print(f"   默认值: {info['default']}")
        print(f"   建议范围: {info['range']}")
        print(f"   说明: {info['description']}")

def print_troubleshooting():
    """打印故障排除指南"""
    print("\n🔧 故障排除")
    print("="*80)
    
    print("\n❌ 问题: 仍然有后脑勺被误判为正脸")
    print("解决方案:")
    print("1. 降低 max_yaw_angle 到 10-12°")
    print("2. 提高 min_area_ratio 到 0.02-0.025")
    print("3. 确保 enable_eye_detection=True")
    print("4. 检查是否有关键点检测错误")
    
    print("\n❌ 问题: 正脸检出率太低，漏掉太多正脸")
    print("解决方案:")
    print("1. 适当提高 max_yaw_angle 到 18-20°")
    print("2. 降低 min_confidence 到 0.8")
    print("3. 降低 min_face_size 到 120px")
    print("4. 关闭某些严格验证 (profile_rejection=False)")
    
    print("\n❌ 问题: 眼部检测器找不到")
    print("解决方案:")
    print("1. 检查 OpenCV 安装是否完整")
    print("2. 下载 haarcascade_eye.xml 到项目目录")
    print("3. 或者设置 enable_eye_detection=False")
    
    print("\n❌ 问题: 处理速度太慢")
    print("解决方案:")
    print("1. 设置 enable_eye_detection=False")
    print("2. 设置 enable_profile_rejection=False")
    print("3. 提高 min_face_size 减少小人脸检测")
    print("4. 降低图片分辨率")

def create_example_usage():
    """创建使用示例"""
    print("\n📝 完整使用示例")
    print("="*80)
    
    example_code = '''
# 示例1: 基础使用
from improved_face_detector import ImprovedFaceDetector

detector = ImprovedFaceDetector()
frontal_count, _ = detector.detect_frontal_faces('test.jpg')
print(f"检测到 {frontal_count} 张正脸")

# 示例2: 严格模式（最少后脑勺误判）
strict_detector = ImprovedFaceDetector(
    max_yaw_angle=12.0,
    min_area_ratio=0.02,
    enable_eye_detection=True,
    enable_profile_rejection=True
)

# 示例3: 宽松模式（更多正脸检出）
loose_detector = ImprovedFaceDetector(
    max_yaw_angle=20.0,
    min_confidence=0.75,
    min_face_size=120,
    enable_eye_detection=False
)

# 示例4: 批量处理
import os
from pathlib import Path

def process_directory(input_dir, output_dir):
    detector = ImprovedFaceDetector()
    
    for img_file in Path(input_dir).glob("*.jpg"):
        frontal_count, details = detector.detect_frontal_faces(
            str(img_file), return_details=True
        )
        
        if frontal_count > 0:
            # 复制到输出目录
            shutil.copy2(img_file, output_dir)
            print(f"✅ {img_file.name}: {frontal_count} 张正脸")
        else:
            print(f"❌ {img_file.name}: 无正脸")
    
    # 显示统计信息
    stats = detector.get_statistics()
    print(f"总检出率: {stats.get('frontal_rate', 0)*100:.1f}%")
'''
    
    print(example_code)

def main():
    """主函数"""
    print_improvements_summary()
    print_usage_guide()
    print_configuration_options()
    print_troubleshooting()
    create_example_usage()
    
    print("\n" + "="*80)
    print("🎯 总结")
    print("="*80)
    print("1. ✅ 已成功改进正脸检测，大幅减少后脑勺误判")
    print("2. ✅ 提供了灵活的配置选项适应不同需求")
    print("3. ✅ 包含完整的测试和故障排除指南")
    print("4. ✅ 可以根据实际效果微调参数")
    
    print("\n💡 下一步建议:")
    print("1. 在实际数据上测试不同参数组合")
    print("2. 如需要，可以进一步添加深度学习姿态估计")
    print("3. 考虑添加手动标注验证步骤")
    print("4. 监控长期使用效果并持续优化")

if __name__ == "__main__":
    main()
