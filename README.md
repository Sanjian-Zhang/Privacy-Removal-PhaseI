# Privacy Removal Phase 1 - Docker Hub Distribution

## 🐳 Available Docker Images

**Docker Hub Repository**: [sanjinn/privacy_removal_phase1](https://hub.docker.com/r/sanjinn/privacy_removal_phase1)

### Available Tags:
- `sanjinn/privacy_removal_phase1:deepprivacy2` - **DeepPrivacy2 Latest (Recommended)** - Face & Full-body anonymization
- `sanjinn/privacy_removal_phase1:face-anon-v3` - Face anonymization v3.0
- `sanjinn/privacy_removal_phase1:garnet` - Text anonymization (CPU optimized)
- `sanjinn/privacy_removal_phase1:dashcam_anonymizer` - Dashcam face & license plate blurring
- `sanjinn/privacy_removal_phase1:deepprivacy` - DeepPrivacy GAN-based face anonymization

## 🚀 Quick Start Guide

### 1. DeepPrivacy2 (Latest - Recommended)

#### Face Anonymization
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/face.py \
  -i /input/image.jpg --output_path /output/result.jpg
```

#### Full-body Anonymization
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/FB_cse.py \
  -i /input/image.jpg --output_path /output/result.jpg --visualize
```

#### Batch Processing
```bash
mkdir -p input output
cp /your/images/* input/

docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  bash -c "find /input -name '*.jpg' -o -name '*.png' | while read img; do python3 anonymize.py configs/anonymizers/face.py -i \"\$img\" --output_path /output/\$(basename \"\$img\"); done"
```

### 2. Text Anonymization (GARNET)
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:garnet
```

### 3. Face Anonymization v3.0
```bash
docker run --runtime=nvidia --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:face-anon-v3
```

### 4. Dashcam Processing
```bash
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

### 5. DeepPrivacy GAN (4K Support)
```bash
docker run --rm \
  -v /path/to/input:/workspace/input \
  -v /path/to/output:/workspace/output \
  -v /path/to/deepprivacy/source:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 2"
```

## 📊 Comparison Table

| Image | Target | GPU Required | Speed | Quality | Unique Feature |
|-------|--------|--------------|-------|---------|----------------|
| `deepprivacy2` | Face + Full-body | Optional | Fast | **State-of-the-Art** | Face + Full-body modes |
| `garnet` | Text regions | No (CPU) | Moderate | Very High | CPU-only operation |
| `face-anon-v3` | Human faces | Yes (12GB+) | Fast | High | Stable version |
| `dashcam_anonymizer` | Faces + License plates | Yes (6GB+) | Very Fast | High | License plate detection |
| `deepprivacy` | Human faces (GAN) | Optional | Medium | Highest | 4K support + GAN generation |



### Available Models
| Model | Quality | Speed | Parameters | Recommended Use |
|-------|---------|-------|------------|-----------------|
| `deep_privacy_v1` | Good | Fast | 46.92M | Quick processing |
| `fdf128_rcnn512` | **Best** | Medium | 47.39M | **Recommended for 4K** |
| `fdf128_retinanet512` | High | Medium | 49.84M | Balanced quality/speed |
| `fdf128_retinanet256` | Medium | Fast | 12.704M | Speed optimized |
| `fdf128_retinanet128` | Basic | Fastest | 3.17M | Low-end hardware |


## 📊 Complete Solution Summary

| Feature | Face Anonymization | Dashcam Anonymizer | GARNET Text | DeepPrivacy GAN | **DeepPrivacy2** |
|---------|-------------------|-------------------|-------------|-----------------|------------------|
| **Target** | Human faces | Faces + License plates | Text regions | Human faces (GAN) | **Faces + Full-body** |
| **Method** | Deep learning blur | YOLOv8 + Gaussian blur | Attention-based inpainting | Generative Adversarial Network | **Latest SOTA Diffusion** |
| **GPU Required** | Yes (12GB+) | Yes (6GB+) | No (CPU optimized) | Optional (6GB+ for acceleration) | **Optional (Auto-detect)** |
| **Quality** | High | High | Very High (95% JPEG) | Highest (Realistic generation) | **State-of-the-Art** |
| **Speed** | Fast | Very Fast | Moderate | Medium (CPU), Fast (GPU) | **Fast (Optimized)** |
| **4K Support** | Yes | Yes | Yes | Optimized for 4K | **Yes** |
| **Memory Management** | Standard | Standard | Standard | Intelligent batching | **Advanced Optimization** |
| **Docker Tag** | `:face-anon-v3` | `:dashcam_anonymizer` | `:garnet` | `:deepprivacy` | **`:deepprivacy2`** |
| **Release Year** | 2023 | 2023 | 2023 | 2022 | **2024** |
| **Unique Features** | - | License plate detection | CPU-only operation | Realistic face generation | **Face + Full-body modes** |
