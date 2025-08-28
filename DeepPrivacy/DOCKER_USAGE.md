# DeepPrivacy Docker 镜像使用说明

## 镜像信息
- **镜像名称**: `sanjinn/privacy_removal_phase1:deepprivacy`
- **大小**: 15.9GB
- **基础环境**: NVIDIA PyTorch 22.08
- **功能**: 4K图片人脸匿名化处理

## 快速开始

### 1. 拉取镜像
```bash
docker pull sanjinn/privacy_removal_phase1:deepprivacy
```

### 2. 处理单张图片
```bash
docker run --rm \
  -v /path/to/your/input:/workspace/input \
  -v /path/to/your/output:/workspace/output \
  -v /path/to/deepprivacy/source:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && python anonymize.py -s /workspace/input/image.jpg -t /workspace/output/anonymized.jpg -m fdf128_rcnn512"
```

### 3. 批量处理图片（推荐）
```bash
docker run --rm \
  -v /path/to/your/input/folder:/workspace/input \
  -v /path/to/your/output/folder:/workspace/output \
  -v /path/to/deepprivacy/source:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 2"
```

## 环境特性

### 已安装的组件
- Python 3.8
- PyTorch 1.13.0a0+d321be6
- OpenCV 4.5.5.64
- Detectron2
- DSFD-Pytorch-Inference
- 所有必要的深度学习库

### 可用模型
- `deep_privacy_v1` (默认)
- `fdf128_rcnn512` (推荐，用于高质量处理)
- `fdf128_retinanet512`
- `fdf128_retinanet256`
- `fdf128_retinanet128`

### 优化特性
- **内存优化**: 支持分批处理，避免内存溢出
- **CPU/GPU兼容**: 自动检测GPU，无GPU时使用CPU
- **4K图片支持**: 专门优化处理高分辨率图片
- **错误恢复**: 支持断点续传，跳过已处理图片

## 使用示例

### 处理4K视频帧
```bash
# 假设您有从视频提取的帧在 /home/user/video_frames
# 输出到 /home/user/anonymized_frames

docker run --rm \
  -v /home/user/video_frames:/workspace/input \
  -v /home/user/anonymized_frames:/workspace/output \
  -v /path/to/deepprivacy:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 3"
```

### GPU加速（如果可用）
```bash
docker run --gpus all --rm \
  -v /path/to/input:/workspace/input \
  -v /path/to/output:/workspace/output \
  -v /path/to/deepprivacy:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && python anonymize.py -s /workspace/input -t /workspace/output -m fdf128_rcnn512"
```

## 注意事项

1. **内存要求**: 处理4K图片时建议至少8GB RAM
2. **存储空间**: 每张4K图片处理后约占用1-2MB额外空间
3. **处理时间**: CPU模式下每张图片约1-2分钟，GPU模式下约10-30秒
4. **批次大小**: 建议4K图片每批处理2-3张，1080P图片每批处理5-10张

## 故障排除

### 内存不足
```bash
# 减少批次大小
./batch_process.sh 1
```

### CUDA不可用警告
这是正常的，镜像会自动切换到CPU模式。

### 权限问题
确保输出目录有写入权限，或使用 `sudo` 运行Docker命令。

## 技术支持

如有问题，请检查：
1. Docker版本 >= 20.10
2. 系统内存 >= 8GB
3. 磁盘可用空间 >= 20GB
4. 输入图片格式为 JPG/PNG/BMP

---
**构建信息**:
- 构建时间: 2025-08-26
- 基础镜像: nvcr.io/nvidia/pytorch:22.08-py3
- DeepPrivacy版本: Latest (带NumPy兼容性修复)
