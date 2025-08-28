#!/bin/bash

# 专用于处理test_dataset的DeepPrivacy2人脸匿名化脚本
# 用法: ./process_test_dataset.sh

INPUT_DIR="/home/zhiqics/sanjian/test_dataset/images/Train"
OUTPUT_DIR="/home/zhiqics/sanjian/output/deep_privacy2_results"
DEEP_PRIVACY2_DIR="/home/zhiqics/sanjian/testdocker/deep_privacy2"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DeepPrivacy2 人脸匿名化批量处理"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "处理模式: 人脸匿名化"

# 统计总图片数
total=$(find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l)
echo "找到 $total 张图片"

# 开始处理时间
start_time=$(date +%s)

echo "=========================================="
echo "开始处理..."

# 运行Docker容器进行批量处理
sudo docker run --rm \
  -v "$INPUT_DIR":/home/zhiqics/input \
  -v "$OUTPUT_DIR":/home/zhiqics/output \
  -v "$DEEP_PRIVACY2_DIR":/home/zhiqics/deep_privacy2 \
  -w /home/zhiqics/deep_privacy2 \
  deep_privacy2 \
  bash -c "
    echo '开始批量人脸匿名化处理...'
    count=0
    success_count=0
    failed_count=0
    
    for img in ../input/*.{jpg,jpeg,png,bmp}; do
        if [ -f \"\$img\" ]; then
            filename=\$(basename \"\$img\")
            # 移除原始前缀，添加匿名化前缀
            output_file=\"../output/anonymized_\$filename\"
            
            # 检查是否已经处理过
            if [ -f \"\$output_file\" ]; then
                echo \"⏭️  跳过已处理: \$filename (\$((++count))/$total)\"
                continue
            fi
            
            echo \"🔄 正在处理: \$filename (\$((++count))/$total)\"
            
            # 处理单张图片，添加错误处理
            if python3 anonymize.py configs/anonymizers/face.py \
                -i \"\$img\" \
                --output_path \"\$output_file\" 2>/dev/null; then
                echo \"✅ 完成: \$filename\"
                ((success_count++))
            else
                echo \"❌ 失败: \$filename\"
                ((failed_count++))
            fi
            
            # 每处理10张图片显示进度
            if [ \$((count % 10)) -eq 0 ]; then
                echo \"📊 进度: \$count/$total (成功: \$success_count, 失败: \$failed_count)\"
            fi
        fi
    done
    
    echo \"===========================================\"
    echo \"批量处理完成！\"
    echo \"总处理数: \$count\"
    echo \"成功: \$success_count\"
    echo \"失败: \$failed_count\"
    echo \"===========================================\"
  "

# 计算处理时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "✅ 所有图片处理完成！"
echo "⏱️  总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "=========================================="

# 显示最终统计
processed_count=$(find "$OUTPUT_DIR" -name "anonymized_*.jpg" | wc -l)
echo "📊 最终统计:"
echo "   - 原始图片: $total 张"
echo "   - 处理完成: $processed_count 张"
echo "   - 成功率: $(echo "scale=1; $processed_count * 100 / $total" | bc)%"
