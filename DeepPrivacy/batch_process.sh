#!/bin/bash

# 分批处理脚本 - 避免内存溢出
# 用法: ./batch_process.sh <输入目录> <输出目录> [批次大小]

INPUT_DIR="/workspace/input"
OUTPUT_DIR="/workspace/output"
BATCH_SIZE=${1:-3}  # 默认每批处理3张图片
MODEL="fdf128_rcnn512"

echo "开始分批处理图片..."
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE 张图片"
echo "使用模型: $MODEL"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取所有图片文件并计数
TOTAL_FILES=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
echo "找到 $TOTAL_FILES 张图片"

BATCH_NUM=1
PROCESSED=0

# 分批处理图片
find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | sort | while IFS= read -r file; do
    filename=$(basename "$file")
    output_file="$OUTPUT_DIR/anonymized_$filename"
    
    # 检查是否已经处理过
    if [ -f "$output_file" ]; then
        echo "跳过已处理的文件: $filename"
        continue
    fi
    
    echo "正在处理: $filename (第 $((PROCESSED + 1)) / $TOTAL_FILES 张)"
    
    # 处理单张图片
    cd /workspace/deepprivacy
    python anonymize.py -s "$file" -t "$output_file" -m "$MODEL"
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功处理: $filename"
    else
        echo "❌ 处理失败: $filename"
    fi
    
    PROCESSED=$((PROCESSED + 1))
    
    # 每处理指定数量的图片后，短暂休息释放内存
    if [ $((PROCESSED % BATCH_SIZE)) -eq 0 ]; then
        echo "完成第 $BATCH_NUM 批 ($BATCH_SIZE 张图片)，休息 3 秒释放内存..."
        sleep 3
        BATCH_NUM=$((BATCH_NUM + 1))
    fi
done

echo "所有图片处理完成！"
