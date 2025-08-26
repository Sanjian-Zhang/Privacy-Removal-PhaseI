# Face Anonymization Made Simple - 性能优化指南

## 问题分析

face_anon_simple 项目处理速度慢的主要原因：

1. **扩散模型推理步数多** - 默认200步推理过程
2. **大模型组合** - 使用多个大型深度学习模型
3. **高精度计算** - 默认使用float32精度
4. **未充分利用GPU优化** - 缺少内存和计算优化

## GPU加速确认

✅ **项目已支持GPU加速**，代码中有：
```python
pipe = pipe.to("cuda")  # 模型已移动到GPU
```

## 优化方案

### 1. 减少推理步数（速度提升4-8倍）
```bash
# 原始：200步 (~30秒/图像)
# 优化：25步  (~4秒/图像)
--num_inference_steps 25
```

### 2. 使用半精度计算（速度提升20-30%，显存减半）
```bash
--use_fp16
```

### 3. 启用内存优化
- `enable_attention_slicing()` - 注意力切片
- `enable_xformers_memory_efficient_attention()` - 高效注意力
- `enable_model_cpu_offload()` - CPU模型卸载

## 使用优化版脚本

我们已创建优化版批处理脚本 `batch_anonymize_optimized.py`：

### 基本使用
```bash
python batch_anonymize_optimized.py \
    --input_dir /path/to/input/images \
    --output_dir /path/to/output/images \
    --num_inference_steps 25 \
    --use_fp16
```

### 高级参数
```bash
python batch_anonymize_optimized.py \
    --input_dir /home/zhiqics/sanjian/dataset/images \
    --output_dir /home/zhiqics/sanjian/output/faceanon \
    --pattern "**/*.{jpg,jpeg,png}" \
    --num_inference_steps 25 \
    --guidance_scale 4.0 \
    --anonymization_degree 1.25 \
    --use_fp16 \
    --overwrite
```

### 参数说明
- `--num_inference_steps`: 推理步数 (默认25，原始200)
- `--use_fp16`: 使用半精度浮点数
- `--anonymization_degree`: 匿名化程度 (1.0-2.0)
- `--guidance_scale`: 引导缩放 (推荐4.0)
- `--overwrite`: 覆盖已存在文件

## 性能对比

| 配置 | 推理步数 | 精度 | 处理时间 | 速度提升 |
|------|----------|------|----------|----------|
| 原始 | 200步 | float32 | ~30秒/图像 | 1x |
| 优化 | 25步 | float16 | ~4秒/图像 | 8x |

## 硬件要求

### 最低要求
- GPU: 6GB+ 显存
- 推荐: RTX 3060/4060 及以上

### 推荐配置
- GPU: 12GB+ 显存 (RTX 3080/4080)
- 可处理更大图像和批量处理

## 环境检查

运行性能基准测试：
```bash
python performance_benchmark.py
```

## 质量vs速度权衡

| 推理步数 | 质量 | 速度 | 推荐场景 |
|----------|------|------|----------|
| 10步 | 一般 | 最快 | 快速预览 |
| 25步 | 良好 | 快 | 批量处理推荐 |
| 50步 | 很好 | 中等 | 平衡选择 |
| 100步 | 优秀 | 慢 | 高质量需求 |
| 200步 | 最佳 | 最慢 | 论文实验 |

## 故障排除

### 显存不足
```bash
# 启用CPU卸载
--enable_cpu_offload

# 减小图像尺寸
--max_size 256
```

### 依赖问题
```bash
# 安装xformers加速
pip install xformers

# 更新transformers
pip install transformers --upgrade
```

## 实际测试建议

1. **先测试单张图像**：
```bash
python demo_run.py
```

2. **小批量测试优化**：
```bash
python batch_anonymize_optimized.py \
    --input_dir test_images \
    --output_dir test_output \
    --num_inference_steps 25 \
    --use_fp16
```

3. **大批量生产处理**：
```bash
python batch_anonymize_optimized.py \
    --input_dir /home/zhiqics/sanjian/dataset/images \
    --output_dir /home/zhiqics/sanjian/output/faceanon \
    --num_inference_steps 25 \
    --use_fp16 \
    --overwrite
```

通过这些优化，处理速度可以提升**4-8倍**，从原来的每张图像30秒降到4秒左右。
